// script.js

// Get the canvas element and its context
// const canvas = document.getElementById('textCanvas');
// const ctx = canvas.getContext('2d');
import { encode, decode } from "https://deno.land/x/gpt_2_3_tokenizer@v0.0.2/mod.js"
const dynamicText = document.getElementById('dynamicText');
const modelStateText = document.getElementById('modelStateText');

const contextMaxLength = 200;
const diplayMaxLength = contextMaxLength;

let vocab = null;





const modelPath = 'gpt.onnx.quantized.onnx';

var model = null;

// Initialize ONNX Runtime
async function initModel() {
    // Load the ONNX model
    console.log("loading model");
    model = await ort.InferenceSession.create(modelPath, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
    console.log("Model loaded");

}



let bufferText = "";

async function updateText(displayedText) {
    // bufferText += newChar;

    
    // if (bufferText.length > 3){
    // Append the new text to the existing content
    dynamicText.textContent += displayedText[0];

    // Scroll the dynamicTextElement to the bottom and add a newline character
    // dynamicText.textContent += '\n';
    dynamicText.scrollTop = dynamicText.scrollHeight;
    

    // Remove the first character if the text is too long
    if (dynamicText.textContent.length > diplayMaxLength) {
        dynamicText.textContent = dynamicText.textContent.slice(1);
    }
    // }
}


async function generateText(prompt) {
    // encode prompt


    
    // console.log(prompt);
    // Convert the prompt to a Tensor
        const input = new ort.Tensor('int32', prompt, [1, prompt.length] );

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
    return nextCharId;
    

}


(async () => {
    modelStateText.textContent = "Loading model";
    await initModel().then(() => {
        modelStateText.textContent = "Model loaded";
    });

    // Generate text from the prompt
    let context = [20];
    let displayedText = "";

    // updateText(displayedText);

    // // Generate a new character every 100ms and update the canvas 
    setInterval(async () => {
        if (displayedText.length < 20) {
            context.push(await generateText(context).catch(err => console.log("Error : "+err)));
            // context += nextChar;
            // console.log(context);
            if (context.length > contextMaxLength) {
                context = context.slice(1);
            }
            displayedText +=  decode([context[context.length-1]])
        }
        // console.log("Displayed text length" + displayedText.length);
        updateText(displayedText).then(() => {
            // console.log("Text updated");
            // console.log(displayedText);
            displayedText = displayedText.slice(1);
            if (displayedText[0] == ' ' && displayedText[1] == '.'){
                displayedText = displayedText.slice(1);
                displayedText = displayedText[0] + ' ' + displayedText[1].toUpperCase() + displayedText.slice(2);
            }
            // console.log("New length" + displayedText.length)
        }
            );
    }, 100);



})();