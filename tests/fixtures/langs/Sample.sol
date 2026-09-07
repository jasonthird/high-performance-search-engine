pragma solidity ^0.8.0;

contract Widget {
    uint public width;

    function render() public view returns (uint) {
        return width * 2;
    }

    function computeTotal(uint[] memory items) public pure returns (uint t) {
        for (uint i = 0; i < items.length; i++) t += items[i];
    }
}
