{
  pkgs ? import (builtins.fetchGit {
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "refs/tags/23.11";
  }) {}, 
  dev ? true,
}:
let py311 = pkgs.python311.withPackages (ps: with ps; [ numpy pandas ]);
    poetryExtras = if dev then [ "coverage" "formatting" "linting" "testsuite" ] else [ ];
    poetryInstallExtras = (
      if poetryExtras == [] then ""
      else pkgs.lib.concatStrings [ " --with " (pkgs.lib.concatStringsSep "," poetryExtras) ]
    );
in
pkgs.mkShell {
  name = "nuclei-vis-napari-env";
  buildInputs = [ pkgs.poetry ];
  shellHook = ''
    # To get this working on the lab machine, we need to modify Poetry's keyring interaction:
    # https://stackoverflow.com/questions/74438817/poetry-failed-to-unlock-the-collection
    # https://github.com/python-poetry/poetry/issues/1917
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
    poetry env use "${py311}/bin/python"
    installcmd="poetry install -vvvv --sync${poetryInstallExtras}"
    echo "Running installation command: $installcmd"
    eval "$installcmd"
    source "$(poetry env info --path)/bin/activate"
  '';
}
