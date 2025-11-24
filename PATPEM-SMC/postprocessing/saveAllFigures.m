function files = saveAllFigures(outDir, varargin)
%SAVEALLFIGURES Save all open figures to a folder in a chosen format.
%
%   FILES = SAVEALLFIGURES(OUTDIR) saves all current figures as .tif (300 dpi)
%   into folder OUTDIR (created if missing). Returns a cell array FILES of paths.
%
%   FILES = SAVEALLFIGURES(OUTDIR, 'Name', Value, ...) supports:
%       'Format'        - file format: 'tif' (default), 'png','jpg','pdf','epsc'
%       'Resolution'    - DPI for raster formats (default 300)
%       'UseFigureName' - true|false, use figure Name as base filename (default true)
%       'IncludeHidden' - true|false, include hidden figures (default true)
%       'Overwrite'     - true|false, overwrite if file exists (default false)
%
%   Examples
%     % Basic usage: save as TIF 300 dpi
%     saveAllFigures('D:\fig_out');
%
%     % Save as PNG 600 dpi without hidden figures
%     saveAllFigures('D:\png_out', 'Format','png','Resolution',600,'IncludeHidden',false);
%
%     % Use numeric indices instead of figure Name, allow overwrite
%     saveAllFigures('D:\out', 'UseFigureName',false, 'Overwrite',true);
%
%   Notes
%     - Prefers exportgraphics (R2020a+). Falls back to print if unavailable or fails.
%     - For vector outputs ('pdf','epsc'), Resolution is ignored by exportgraphics.
%
%   See also: exportgraphics, print, findall

% -------- Parse inputs --------
p = inputParser;
validFmt = @(s) ischar(s) || (isstring(s) && isscalar(s));
addRequired(p,'outDir',@(s) ischar(s)||isstring(s));
addParameter(p,'Format','tif',validFmt);
addParameter(p,'Resolution',300,@(x) isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p,'UseFigureName',true,@islogical);
addParameter(p,'IncludeHidden',true,@islogical);
addParameter(p,'Overwrite',false,@islogical);
parse(p,outDir,varargin{:});
outDir = char(p.Results.outDir);
fmt    = lower(char(p.Results.Format));
dpi    = p.Results.Resolution;
useName= p.Results.UseFigureName;
inclH  = p.Results.IncludeHidden;
overwrite = p.Results.Overwrite;

% Supported formats mapping (for print)
printMap = struct('tif','-dtiff','png','-dpng','jpg','-djpeg','jpeg','-djpeg', ...
                  'pdf','-dpdf','eps','-depsc','epsc','-depsc');

if ~isfield(printMap, fmt)
    error('Unsupported format "%s". Use tif|png|jpg|pdf|epsc.', fmt);
end

% Ensure output directory exists
if ~exist(outDir,'dir'), mkdir(outDir); end

% Get figure handles
if inclH
    figs = findall(0,'Type','figure'); % include hidden figures
else
    figs = findobj('Type','figure');   % exclude hidden figures
end
if isempty(figs)
    warning('No open figures found.');
    files = {};
    return
end

% Sort by figure number
nums = get(figs, 'Number');
if iscell(nums), nums = cell2mat(nums); end
[~,idx] = sort(nums);
figs = figs(idx);

% Save one by one
files = cell(numel(figs),1);
for k = 1:numel(figs)
    f = figs(k);

    % Build base filename
    if useName && ~isempty(f.Name)
        base = matlab.lang.makeValidName(f.Name);
    else
        base = sprintf('figure_%d', f.Number);
    end
    file = fullfile(outDir, [base '.' fmt]);

    % Handle duplicate names
    if ~overwrite
        c = 1;
        while exist(file,'file')
            file = fullfile(outDir, sprintf('%s_%02d.%s', base, c, fmt));
            c = c + 1;
        end
    end

    % Save (prefer exportgraphics)
    didSave = false;
    try
        switch fmt
            case {'pdf','eps','epsc'}         % vector formats
                exportgraphics(f, file);
            otherwise                          % raster formats
                exportgraphics(f, file, 'Resolution', dpi);
        end
        didSave = true;
    catch
        % Fallback to print
        try
            set(f,'PaperPositionMode','auto');
            printOpt = printMap.(fmt);
            [pth,nam,~] = fileparts(file);
            if any(strcmp(fmt, {'pdf','eps','epsc'}))
                % print ignores -r for vector; still harmless to pass
                print(f, fullfile(pth, nam), printOpt);
            else
                print(f, fullfile(pth, nam), printOpt, sprintf('-r%d', dpi));
            end
            didSave = true;
        catch ME
            warning('Failed to save "%s": %s', file, ME.message);
            didSave = false;
        end
    end

    if didSave
        files{k} = file;
    else
        files{k} = '';
    end
end

% Remove failed entries
files = files(~cellfun(@isempty, files));

% Summary
fprintf('Saved %d figure(s) to: %s\n', numel(files), outDir);
end
