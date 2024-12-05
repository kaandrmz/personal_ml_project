<template>
    <div style="width: 100vw; height: 100vh; position: relative;">
        <BaklavaEditor :view-model="baklava" />
        <div style="position: absolute; top: 10px; right: 10px; z-index: 1000; display: flex; flex-direction: column; gap: 10px;">
            <button @click="postGraph">Execute</button>
            <div v-if="generatedCode" class="code-container">
                <div class="code-header">
                    <span>Generated Code</span>
                    <button @click="copyCode" class="copy-button">
                        {{ copied ? 'Copied!' : 'Copy' }}
                    </button>
                </div>
                <pre><code ref="codeBlock" class="language-python">{{ generatedCode }}</code></pre>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { BaklavaEditor, useBaklava } from "@baklavajs/renderer-vue";
import "@baklavajs/themes/dist/syrup-dark.css";
import linearNode from "./linearNode";
import conv2dNode from "./conv2dNode";
import axios from 'axios';
import { ref, onMounted, nextTick, watch } from 'vue';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github-dark.css';

hljs.registerLanguage('python', python);

const baklava = useBaklava();
baklava.editor.registerNodeType(linearNode);
baklava.editor.registerNodeType(conv2dNode);

const generatedCode = ref('');
const codeBlock = ref(null);
const copied = ref(false);

// Watch for changes in generatedCode and re-apply highlighting
watch(generatedCode, async () => {
    await nextTick();
    
    if (codeBlock.value) {
        // Remove the data-highlighted attribute
        codeBlock.value.removeAttribute('data-highlighted');
        // Remove any existing highlighting classes
        codeBlock.value.className = 'language-python';
        codeBlock.value.textContent = generatedCode.value;
        
        // Force a new highlight
        hljs.highlightElement(codeBlock.value);
    }
});
onMounted(() => {
    if (generatedCode.value && codeBlock.value) {
        hljs.highlightElement(codeBlock.value);
    }
});

function copyCode() {
    if (generatedCode.value) {
        navigator.clipboard.writeText(generatedCode.value);
        copied.value = true;
        setTimeout(() => {
            copied.value = false;
        }, 2000);
    }
}

function saveGraph() {
  // Save the current graph (nodes, connections, etc.)
  const graphData = baklava.editor.save();
  return graphData;
}

// Define your execute function
async function postGraph() {
    const graphData = saveGraph();
    try {
        const response = await axios.post('http://localhost:5000/api/saveGraph', graphData, {
            headers: {
                'Content-Type': 'application/json',
            }
        });
        generatedCode.value = response.data.code;
    }
    catch (error) {
        console.error('Error sending data to Flask:', error);
    }
}
</script>

<style scoped>
.code-container {
    background-color: #0d1117;
    border-radius: 8px;
    width: 400px;
    max-height: 600px;
    overflow: auto;
    border: 1px solid #30363d;
}

.code-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background-color: #161b22;
    border-bottom: 1px solid #30363d;
}

.code-header span {
    color: #c9d1d9;
    font-size: 14px;
}

.copy-button {
    background-color: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 12px;
    transition: all 0.2s;
}

.copy-button:hover {
    background-color: #30363d;
    border-color: #8b949e;
}

button {
    cursor: pointer;
}

pre {
    margin: 0;
    padding: 16px;
}

code {
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 14px;
}

:deep(.hljs) {
    background-color: transparent !important;
    padding: 0 !important;
}
</style>