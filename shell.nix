{
  pkgs ? import (builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "refs/tags/23.11";
  }) {}, 
  dev ? true,
}:
let 
  pyenv = pkgs.python311.withPackages (pp: with pp; [ pip wheel ]);
  pipInstallExtras = if dev then "\"[coverage,formatting,linting,testsuite]\"" else "";
in
pkgs.mkShell {
  name = "nuclei-vis-napari-env";
  buildInputs = [ pyenv ];
  shellHook = ''
    [[ -d .venv ]] || python3.11 -m venv .venv
    source .venv/bin/activate
    installcmd='pip install -e .${pipInstallExtras}'
    echo "Running installation command: $installcmd"
    eval "$installcmd"
  '';
}
