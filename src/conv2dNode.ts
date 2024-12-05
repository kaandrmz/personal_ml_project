import { defineNode, NodeInterface, IntegerInterface, CheckboxInterface } from "baklavajs";

export default defineNode({
    type: "conv2dNode",
    inputs: {
        input: () => new NodeInterface("Input", 0),
        in_channels: () => new IntegerInterface("in_channels", 1, 0).setPort(false),
        out_channels: () => new IntegerInterface("out_channels", 1, 0).setPort(false),
        kernel_size: () => new IntegerInterface("kernel_size", 3, 1).setPort(false),
    },
    outputs: {
        output: () => new NodeInterface("Output", 0),
    },
});
