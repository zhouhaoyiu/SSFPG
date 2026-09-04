function build_SSFPG_gram_rust
root=fileparts(mfilename('fullpath'));
builddir=fullfile(tempdir,'ssfpg_gram_rust_build');
if ~exist(builddir,'dir'),mkdir(builddir);end
if ispc
    lib=fullfile(builddir,'ssfpg_gram_core.lib');
    cflags='COMPFLAGS=$COMPFLAGS /O2 /DNDEBUG';
else
    lib=fullfile(builddir,'libssfpg_gram_core.a');
    cflags='COPTIMFLAGS=$COPTIMFLAGS -O3 -DNDEBUG';
end
cmd=sprintf('rustc --crate-type staticlib -C opt-level=3 -C target-cpu=native -C panic=abort "%s" -o "%s"', ...
    fullfile(root,'ssfpg_gram_core.rs'),lib);
[status,message]=system(cmd);
if status~=0,error('SSFPG_gram_rust:rustc','%s',message);end
mex('-R2018a','-silent',cflags, ...
    fullfile(root,'ssfpg_gram_rust_mex.c'),lib,'-output',fullfile(root,'SSFPG_gram_rust_mex'));
fprintf('Built %s\n',fullfile(root,['SSFPG_gram_rust_mex.' mexext]));
end
