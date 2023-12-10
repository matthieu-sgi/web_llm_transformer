// script.js

// Get the canvas element and its context
// const canvas = document.getElementById('textCanvas');
// const ctx = canvas.getContext('2d');
const dynamicText = document.getElementById('dynamicText');

const contextMaxLength = 200;
const diplayMaxLength = contextMaxLength;

let vocab = null;





const modelPath = 'gpt.onnx';

// Initialize ONNX Runtime
async function initModel() {
    // Load the ONNX model
    model = await ort.InferenceSession.create(modelPath);
    console.log("Model loaded");

}

// let xPos = 10; // Initial x-position for drawing text
// let yPos = 50; // Initial y-position for drawing text

// async function updateCanvas(text) {
//     ctx.clearRect(0, 0, canvas.width, canvas.height);
//     ctx.font = '30px serif';

//     // Split text into lines based on canvas width
//     const lines = [];
//     const maxWidth = canvas.width - 20; // Adjust as needed
//     let currentLine = '';

//     for (let i = 0; i < text.length; i++) {
//         const testLine = currentLine + text[i];
//         const testWidth = ctx.measureText(testLine).width;
//         if (testWidth > maxWidth && i > 0) {
//             lines.push(currentLine);
//             currentLine = text[i];
//         } else {
//             currentLine = testLine;
//         }
//     }

//     lines.push(currentLine);

//     // Draw text lines on the canvas
//     for (const line of lines) {
//         ctx.fillText(line, xPos, yPos);
//         yPos += 40; // Adjust the line height as needed
//     }

//     // Slide text when it touches the end of the canvas
//     if (yPos > canvas.height) {
//         yPos = 50; // Reset y-position
//     }
// }

let bufferText = "";

async function updateText(newChar) {
    bufferText += newChar;

    if (bufferText[0] == ' ' && bufferText[1] == '.'){
        bufferText = bufferText.slice(1);
        bufferText = bufferText[0] + ' ' + bufferText[1].toUpperCase();
    }

    if (bufferText.length > 3){
    // Append the new text to the existing content
    dynamicText.textContent += newChar;

    // Scroll the dynamicTextElement to the bottom and add a newline character
    // dynamicText.textContent += '\n';
    dynamicText.scrollTop = dynamicText.scrollHeight;
    

    // Remove the first character if the text is too long
    if (dynamicText.textContent.length > diplayMaxLength) {
        dynamicText.textContent = dynamicText.textContent.slice(1);
    }
    bufferText = bufferText.slice(1);
    }
}


async function generateText(prompt) {
    // TODO: Optimization : Avoid encoding the prompt at each step
    // encode prompt
    // console.log("Generating text...");
    console.log(prompt);
    prompt = prompt.toLowerCase();
    prompt = prompt.split('').map(c => vocab.char_to_id[c]);
    console.log(prompt);
    // Convert the prompt to a Tensor
    // const input = ort.tensor(prompt, [1, prompt.length], 'int32');
    const input = new ort.Tensor('int64', prompt, [1, prompt.length] );

    // Run inference
    const output = await model.run({'input':input});
    // console.log(output);
    // Get the predicted next character
    const output_data = output.output.data;

    const sum = output_data.reduce((a, b) => a + Math.exp(b), 0);
    const normalized = output_data.map(x => Math.exp(x) / sum);
    

    //! Sampling from the distribution
    // Cumulative distribution function
    const cdf = [];
    let sum2 = 0;
    for (let i = 0; i < normalized.length; i++) {
        sum2 += normalized[i];
        cdf.push(sum2);
    }
    // console.log("CDF:");
    // console.log(cdf);

    // Sample from the CDF

    const r = Math.random();
    // console.log("r:");
    // console.log(r);

    let nextCharId = 0;
    for (let i = 0; i < cdf.length; i++) {
        if (r < cdf[i]) {
            nextCharId = i;
            break;
        }
    }

    // console.log("Next character id:");
    // console.log(nextCharId);

    const nextChar = vocab.id_to_char[nextCharId];

    // console.log("Next character:");
    // console.log(nextChar);

    // Update the prompt
    return nextChar;
    
    

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
    let context = 'hello';
    let displayedText = context;

    updateText(displayedText);

    // Generate a new character every 100ms and update the canvas 
    setInterval(async () => {
        // console.log("Contextsize : " + context.length);
        nextChar = await generateText(context).catch(err => console.log("Error : "+err));
        context += nextChar;
        if (context.length > contextMaxLength) {
            context = context.slice(1);
        }
        
        updateText(nextChar);
    }, 100);



})();