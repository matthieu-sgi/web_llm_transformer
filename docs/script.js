// script.js

// Get the canvas element and its context
const canvas = document.getElementById('textCanvas');
const ctx = canvas.getContext('2d');


// async function loadVocab() {
//     console.log("Loading vocab...");
//     const response = await fetch("vocab.json");
//     const data = await response.json();

//     return data;
// }

// fetch vocab from json file and put it in a global variable

let vocab = null;

// async function loadVocab() {
//     console.log("Loading vocab...");
//     fetch("vocab.json")
//         .then(response => response.json())
//         .then(data => {
//             vocab = data;
//             console.log("Vocab loaded");
//         });
// }








const modelPath = 'gpt.onnx';

// Initialize ONNX Runtime
async function initModel() {
    // Load the ONNX model
    model = await ort.InferenceSession.create(modelPath);
    console.log("Model loaded");

}


// Generate text from a prompt

async function generateText(prompt = 'hello') {
    // encode prompt
    console.log("Generating text...");
    console.log(prompt);
    prompt = prompt.toLowerCase();
    prompt = prompt.split('').map(c => vocab.char_to_id[c]);
    console.log(prompt);
    // Convert the prompt to a Tensor
    // const input = ort.tensor(prompt, [1, prompt.length], 'int32');
    const input = new ort.Tensor('int64', prompt, [1, prompt.length] );

    // Run inference
    const output = await model.run({'input':input});
    console.log(output);
    // Get the predicted next character
    const nextChar = output.output.data;

    const sum = nextChar.reduce((a, b) => a + Math.exp(b), 0);
    const normalized = nextChar.map(x => Math.exp(x) / sum);
    
    // Return the next character
    console.log("Next character:");
    console.log(normalized);

    

}

async function loadVocab() {

fetch("vocab.json")
    .then(response => response.json())
    .then(data => {
        vocab = data;
        console.log("Vocab loaded");
        console.log("Vocab:");
        console.log(vocab);

    });
}


(async () => {
    await loadVocab();

    await initModel();

    // Generate text from the prompt
    let prompt = 'hello';
    generateText(prompt);

})();