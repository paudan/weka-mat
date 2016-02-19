function directory = script_dir
path = mfilename('fullpath');
idx = regexp(path,'\\');
[m,n]=size(idx);
directory = path(1:idx(1,n));
