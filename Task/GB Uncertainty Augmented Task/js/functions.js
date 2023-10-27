function fixation(c) { 
    //generates fixation by drawing on HTML canvas.
    //function takes the canvas(c) as input and generates the figure
    var context = c.getContext("2d"); //Name of the canvas
    context.moveTo(240, 250); //start of horizontal line
    context.lineTo(280, 250); //end of horizontal line
    context.moveTo(260, 230); //start of vertical line
    context.lineTo(260, 270); //end of vertical line
    context.filter = 'contrast(1)'; //keeps the contrast constant
    context.lineWidth = 3; // width of the lines in the fixation cross
    context.stroke(); // draws the images on the canvas 
};

function gabor(c) { 
    // function takes the canvas as input
    //generates 2 gabors left and right to the fixation
    var context = c.getContext("2d"); //Name of the canvas
    const my_gradient = context.createLinearGradient(0,0,200,0); const my_gradient1 = context.createLinearGradient(225,0,425,0); //two gradients for two gabors
    const bands = 10; const colors = ["#000", "#FFF"]; // setting the color for the bands
    const stops = bands * colors.length; //number of layers that you want to color in the patch
    var pos = 0; var pos1 = 0;
    while (pos <= stops) { //colors each band of the patch with white and black.
        my_gradient1.addColorStop(pos / stops, colors[pos % colors.length]);
        pos++;
    }
    while (pos1 <= stops) {
        my_gradient.addColorStop(pos1 / stops, colors[pos1 % colors.length]);
        pos1++;
    }
    context.fillStyle = my_gradient; // defining the gradient for the left Gabor
    context.filter = 'contrast('+ jsPsych.currentTrial().contrast_left +')' //contrast level randomly generated after every trial
    context.fillRect(100,200,100,100); //co-ordinates of the left patch
    context.fillStyle = my_gradient1; // defining the gradient for the right Gabor
    context.filter = 'contrast('+ jsPsych.currentTrial().contrast_right+')'
    context.fillRect(325,200,100,100); //co-ordinates of the left patch
    context.stroke() // drawing the two patches
    context.moveTo(240, 250); context.lineTo(280, 250); //horizontal line
    context.moveTo(260, 230); context.lineTo(260, 270); //vertical line
    context.filter = 'contrast(1)'; context.lineWidth = 3;
    context.stroke(); // drawing the fixation cross
};

function randbet(min, max) { 
    //randbet takes two inputs and generates a randomly number between the min and max number. 
    // min and max are the upper and lower bound.
    return Math.random() * (max - min) + min; 
};

function shuffle(array) {
    // takes a list of elements as its input
    // shuffles the array and changes the location of its elements
    var len = array.length, temp, index; //length of the array
    while (len > 0) { // While there are elements in the array
        index = Math.floor(Math.random() * len); // Pick a random index
        len--; // Decrease len by 1
        temp = array[len]; // And swap the last element with it
        array[len] = array[index];
        array[index] = temp;
    }
    return array;
};
// Reference: https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
