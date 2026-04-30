function pe = setupMonitorioPython(varargin)
%SETUPMONITORIOPYTHON Configure pyenv for the Monitorio Python tools.
%
%   pe = setupMonitorioPython() locates the Monitorio repo's venv
%   (alongside this .m file at <repo>/venv/), points pyenv at it in
%   OutOfProcess mode if not already configured that way, and ensures
%   <repo>/Source is on Python's sys.path so 'decode_sync_tags' and
%   friends can be imported. Returns the configured pyenv object.
%
%   pe = setupMonitorioPython('VenvPath', P) uses a different venv at P.
%
%   Idempotent: a second call with the same venv is a no-op.
%
%   Notes
%   -----
%   * MATLAB's pyenv must be set BEFORE the first Python call in a
%     given MATLAB session. If Python has already been loaded with a
%     different interpreter, restart MATLAB before calling this.
%   * Uses ExecutionMode='OutOfProcess'. This avoids library conflicts
%     when MATLAB and Python load incompatible BLAS/Qt/etc. versions.
%   * On Windows the venv interpreter is venv\Scripts\python.exe.
%     On Linux/macOS it is venv/bin/python.
%
%   Example
%   -------
%       setupMonitorioPython();
%       result = decodeSyncTags(samples, 50000, 'tagged.mp4', 'cal.json');

p = inputParser;
p.addParameter('VenvPath', '', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});

% Default venv path: <dir-containing-this-file>/../venv
sourceDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(sourceDir);
venvPath = char(p.Results.VenvPath);
if isempty(venvPath)
    venvPath = fullfile(repoRoot, 'venv');
end

if ispc
    pythonExe = fullfile(venvPath, 'Scripts', 'python.exe');
else
    pythonExe = fullfile(venvPath, 'bin', 'python');
end
if ~isfile(pythonExe)
    error('setupMonitorioPython:venvNotFound', ...
        'No Python interpreter at %s. Create the venv with: "python -m venv %s" then "pip install -r %s".', ...
        pythonExe, venvPath, fullfile(repoRoot, 'requirements.txt'));
end

% Configure pyenv only if it isn't already pointing at this interpreter,
% or its execution mode isn't OutOfProcess.
pe = pyenv;
needsSetup = ~strcmp(char(pe.Executable), pythonExe) ...
          || ~strcmp(char(pe.ExecutionMode), 'OutOfProcess');
if needsSetup
    if pe.Status == "Loaded"
        error('setupMonitorioPython:pythonAlreadyLoaded', ...
            ['Python is already loaded with interpreter %s in %s mode. ', ...
             'pyenv cannot be reconfigured after Python is loaded -- ', ...
             'restart MATLAB and call setupMonitorioPython again before ', ...
             'any other Python calls.'], ...
            char(pe.Executable), char(pe.ExecutionMode));
    end
    pe = pyenv('Version', pythonExe, 'ExecutionMode', 'OutOfProcess');
end

% Make sure <repo>/Source is on Python's sys.path so
% `import decode_sync_tags` resolves. Only adds it if not already present.
sysPath = py.sys.path;
if count(sysPath, sourceDir) == 0
    insert(sysPath, int32(0), sourceDir);
end
end
