{
  # The latest stable release of nixpkgs. A tarball downloads much faster than a clone of the
  # whole repository; to update it, change the URL and get the new hash with
  # `nix-prefetch-url --unpack <url>`.
  pkgs ? import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/refs/tags/26.05.tar.gz";
    sha256 = "0am8xx09fx5yf2p0wb001v0jx1g5hrfb76h4r37xph378jgk7pcr";
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
    installcmd="poetry sync -vvvv${poetryInstallExtras}"
    echo "Running installation command: $installcmd"
    eval "$installcmd"
    source "$(poetry env info --path)/bin/activate"
  '';
}
