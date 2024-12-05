import { defineNode, NodeInterface, NumberInterface, SelectInterface, IntegerInterface } from "baklavajs";

export default defineNode({
    type: "firstNode",
    inputs: {
        number1: () => new IntegerInterface("Number", 1, 0),
        number2: () => new NumberInterface("Number", 10),
        operation: () => new SelectInterface("Operation", "Add", ["Add", "Subtract"]).setPort(false),
    },
    outputs: {
        output: () => new NodeInterface("Output", 0),
    },
});
