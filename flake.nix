{
  description = "Tokenization_Text_to_Numbers - project-local Python env (uv-managed), isolated from global home.nix/configuration.nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.python312
            pkgs.uv
            pkgs.git
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export UV_PYTHON="${pkgs.python312}/bin/python3"

            # --frozen so lock drift fails loudly instead of silently rewriting uv.lock.
            # Run `uv lock` deliberately when dependencies change.
            uv sync --frozen

            echo "Tokenization_Text_to_Numbers dev shell ready (uv-managed, project-local)."
            echo "Python: $(python3 --version 2>/dev/null || echo 'not on PATH until venv is activated')"
            echo "Activate: source .venv/bin/activate"
          '';
        };
      });
}
