{ pkgs ? import <nixpkgs> {} }:
rec {
  computeTotal = items: builtins.foldl' (a: b: a + b) 0 items;
  render = width: pkgs.lib.strings.replicate width " ";
}
