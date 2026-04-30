function result = decodeSyncTags(samples, sampleRate, videoPath, calibrationPath, varargin)
%DECODESYNCTAGS MATLAB wrapper for the Python decode.decode_sync_tags.
%
%   result = decodeSyncTags(samples, sampleRate, videoPath, calibrationPath)
%   decodes per-frame sample indices from the photodiode recording in
%   `samples` (an n_channels-by-n_samples numeric matrix) using the
%   tagged video at `videoPath` and the calibration JSON at
%   `calibrationPath`. Returns a MATLAB struct with the decoded frame
%   table and the same diagnostics the Python decoder returns.
%
%   Channel order in `samples` must match the photodiodes list in the
%   calibration JSON (same order calibrate.py wrote them, which is the
%   physical AI-pin order).
%
%   Optional name-value parameters mirror the Python API:
%
%     'Scale'              numeric or char/string (default 1.0).
%                          Multiplier converting raw sample units to
%                          volts. Pass 'intan_aux' (= 0.0000374) when
%                          the recording is raw Intan ADC steps from
%                          the headstage auxiliary inputs; 'volts' (1.0)
%                          when the recording is already in volts (e.g.
%                          NI-DAQmx default). See SCALE_PRESETS in
%                          decode_sync_tags.py for the full list.
%     'DebounceFraction'   numeric (default 0.25). Glitch-rejection
%                          window expressed as a fraction of one frame
%                          interval.
%     'SyncBitOverride'    [] (default), true, or false. When set,
%                          forces the sync_bit interpretation regardless
%                          of what the tagger's sidecar says.
%     'OutputPath'         '' (default) or path. If non-empty, also
%                          writes the result as a CSV with a metadata
%                          header.
%     'Metadata'           '' (default) or string. Extra string baked
%                          into the CSV header. Multi-line OK.
%     'Verbose'            integer (default 0). 0 silent, 1 one-line
%                          summary to stderr, 2 also dumps thresholds,
%                          SNR, and per-transition unwrap accounting.
%     'AutoSetup'          logical (default true). When true, calls
%                          setupMonitorioPython before each decode so
%                          pyenv is configured.
%
%   The returned struct has fields:
%
%     frameTable           n-by-2 double matrix; columns are
%                          (frameNumber, sampleIndex). frameNumber is
%                          1-indexed; sampleIndex is 0-based into the
%                          original `samples` array, matching Python.
%                          Add 1 to get a MATLAB 1-based index.
%     fps                  double, frame rate read from videoPath.
%     sampleRate           double, the sample_rate you passed in.
%     syncBit              logical, the resolved sync_bit setting.
%     cycle                double, frame-number wrap modulus.
%     nPDs                 double, number of photodiodes.
%     thresholdsV          1-by-nPDs double, per-channel detected
%                          thresholds in volts (post-scale).
%     segmentStartSample   double, 0-based start of the detected video
%                          segment in `samples`.
%     segmentEndSample     double, 0-based exclusive end.
%     warnings             N-by-1 string array; soft diagnostics. Hard
%                          errors come back as MATLAB exceptions
%                          re-raised from the Python side.
%
%   Example
%   -------
%       % NI-DAQmx recording in volts:
%       result = decodeSyncTags(samples, 50000, 'tagged.mp4', 'cal.json');
%
%       % Intan recording in raw ADC steps; write a CSV; verbose decoder:
%       result = decodeSyncTags(samples, 30000, 'tagged.mp4', 'cal.json', ...
%           'Scale', 'intan_aux', ...
%           'OutputPath', 'frames.csv', ...
%           'Metadata', sprintf('%s rig A trial 7', datestr(now, 'yyyy-mm-dd')), ...
%           'Verbose', 1);
%
%       % Convert to MATLAB 1-based indices:
%       sampleIndex1Based = result.frameTable(:, 2) + 1;

% --- argument parsing --------------------------------------------------
p = inputParser;
p.addRequired('samples', @(x) isnumeric(x) && ~isscalar(x) && ndims(x) == 2);
p.addRequired('sampleRate', @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addRequired('videoPath', @(x) ischar(x) || isstring(x));
p.addRequired('calibrationPath', @(x) ischar(x) || isstring(x));
p.addParameter('Scale', 1.0, @(x) (isnumeric(x) && isscalar(x)) || ischar(x) || isstring(x));
p.addParameter('DebounceFraction', 0.25, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('SyncBitOverride', [], @(x) isempty(x) || islogical(x));
p.addParameter('OutputPath', '', @(x) ischar(x) || isstring(x));
p.addParameter('Metadata', '', @(x) ischar(x) || isstring(x));
p.addParameter('Verbose', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('AutoSetup', true, @islogical);
p.parse(samples, sampleRate, videoPath, calibrationPath, varargin{:});
opts = p.Results;

if opts.AutoSetup
    setupMonitorioPython();
end

% --- type conversion ---------------------------------------------------
np = py.importlib.import_module('numpy');

% MATLAB → numpy. Force float64 since downstream Python expects it
% (cast happens upstream in decode_sync_tags too, but explicit here is
% cheaper than letting numpy figure it out from a non-double matrix).
samplesPy = np.asarray(double(opts.samples));

% Scale: numeric or named preset.
if ischar(opts.Scale) || isstring(opts.Scale)
    scalePy = py.str(char(opts.Scale));
else
    scalePy = py.float(opts.Scale);
end

% Optional args: empty MATLAB values map to Python None.
if isempty(opts.SyncBitOverride)
    syncBitPy = py.None;
else
    syncBitPy = py.bool(logical(opts.SyncBitOverride));
end
if strlength(string(opts.OutputPath)) == 0
    outputPathPy = py.None;
else
    outputPathPy = py.str(char(opts.OutputPath));
end
if strlength(string(opts.Metadata)) == 0
    metadataPy = py.None;
else
    metadataPy = py.str(char(opts.Metadata));
end

% --- call Python -------------------------------------------------------
% Reload the module each call so the user picks up edits to
% decode_sync_tags.py during development without having to restart
% MATLAB. Cheap.
mod = py.importlib.import_module('decode_sync_tags');
py.importlib.reload(mod);

pyResult = mod.decode_sync_tags( ...
    samplesPy, ...
    py.float(opts.sampleRate), ...
    py.str(char(opts.videoPath)), ...
    py.str(char(opts.calibrationPath)), ...
    pyargs( ...
        'scale', scalePy, ...
        'debounce_fraction', py.float(opts.DebounceFraction), ...
        'sync_bit_override', syncBitPy, ...
        'output_path', outputPathPy, ...
        'metadata', metadataPy, ...
        'verbose', py.int(opts.Verbose) ...
    ));

% --- result conversion -------------------------------------------------
result = struct();
% frame_table: numpy int64 (n, 2) -> MATLAB double (n, 2). Stay 0-based
% to match the Python convention; the docstring tells the user to
% +1 if they want MATLAB-style indexing.
result.frameTable        = double(pyResult.frame_table);
result.fps               = double(pyResult.fps);
result.sampleRate        = double(pyResult.sample_rate);
result.syncBit           = logical(pyResult.sync_bit);
result.cycle             = double(pyResult.cycle);
result.nPDs              = double(pyResult.n_pds);
result.thresholdsV       = cellfun(@double, cell(pyResult.thresholds_v));
result.segmentStartSample = double(pyResult.segment_start_sample);
result.segmentEndSample  = double(pyResult.segment_end_sample);
% warnings_ is a Python list of str; convert to MATLAB string array.
warningsCell = cell(pyResult.warnings_);
if isempty(warningsCell)
    result.warnings = string.empty(0, 1);
else
    result.warnings = string(cellfun(@(s) string(s), warningsCell, 'UniformOutput', true)');
end
end
