import { defineNode, NodeInterface, IntegerInterface, CheckboxInterface } from "baklavajs";

export default defineNode({
    type: "linearNode",
    inputs: {
        input: () => new NodeInterface("Input", 0), // or should it be null as default?
        in_features: () => new IntegerInterface("in_features", 1, 0).setPort(false),
        out_features: () => new IntegerInterface("out_features", 1, 0).setPort(false),
    },
    outputs: {
        output: () => new NodeInterface("Output", 0),
    },
});
