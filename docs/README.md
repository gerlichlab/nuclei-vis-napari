## Documentation
Here you will find both a [user guide](./using-the-plugin.md) and [development documentation](./development.md).

### Installation and environment
This project uses a Nix shell to provide a development environment. 
To use this plugin, you have at least two options.

__Nix shell (preferred)__

If you've installed the [Nix package manager](https://nixos.org/download/) installed, then from the this project's [root folder](../), you can simply launch the Nix shell with `nix-shell`, and you should have everything you need ready to go. 
Note that in general, we use this on Mac machines with relatively new processors. 
If something doesn't work for you, and especially if you're using this on different hardware, please let us know by opening a ticket on our [issue tracker](https://github.com/gerlichlab/nuclei-vis-napari/issues). 
Please open a ticket even if you're able to resolve the problem yourself, as it will help us define the environment in a way that's compatible with a wider variety of hardwares.
If you'd like, you may even open a pull request to suggest an improvement to our environment definition.

__Virtual environment (e.g. `virtualenv`)__

If you're not inclined to use Nix, ensure that you've installed `virtualenv` (e.g., `brew install virtualenv` on a Mac), and that you have Python available (e.g., `brew install python@3.11`). Then you may create and activate a virtual environment, into which you may install this project. Simply run `pip install .` from this project's [root folder](../).
