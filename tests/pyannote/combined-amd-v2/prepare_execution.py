"""Native inventory selection reused by the new payload preparation."""
def selected_native_files(files):
    selected = {}
    packages = ('numpy', 'scipy', 'torch', 'torchaudio', 'onnxruntime', 'pyannote', 'einops', 'pandas', 'sortedcontainers', 'psutil')
    for name, wanted in files.items():
        segment = name.split('/site-packages/', 1)[-1] if '/site-packages/' in name else name.split('/python/', 1)[-1] if '/python/' in name else ''
        package = segment.split('/')[0]
        if any(package == p or package.startswith(p+'.') or package.startswith(p+'-') or package.startswith(p+'_') for p in packages):
            selected[name] = wanted
    return selected

