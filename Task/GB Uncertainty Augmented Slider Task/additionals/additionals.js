function fixation(c) {
    //generates fixation by drawing on HTML canvas.
    //function takes the canvas(c) as input and generates the figure
    var context = c.getContext("2d"); //Name of the canvas
    context.moveTo(230, 275); context.lineTo(270, 275); //horizontal line
    context.moveTo(251, 255); context.lineTo(251, 295); //vertical line
    //context.moveTo(240, 250); //start of horizontal line
    //context.lineTo(280, 250); //end of horizontal line
    //context.moveTo(260, 230); //start of vertical line
    //context.lineTo(260, 270); //end of vertical line
    context.filter = 'contrast(1)'; //keeps the contrast constant
    context.lineWidth = 3; // width of the lines in the fixation cross
    context.stroke(); // draws the images on the canvas
}

function gabor(c) {
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
    context.fillRect(100,220,100,100); //co-ordinates of the left patch
    context.fillStyle = my_gradient1; // defining the gradient for the right Gabor
    context.filter = 'contrast('+ jsPsych.currentTrial().contrast_right +')'
    context.fillRect(305,220,100,100); //co-ordinates of the right patch
    context.stroke() // drawing the two patches

    context.beginPath();
    context.strokeStyle = "#000000";
    context.moveTo(230, 275); context.lineTo(270, 275); //horizontal line
    context.moveTo(251, 255); context.lineTo(251, 295); //vertical line
    context.filter = 'contrast(1)'; context.lineWidth = 3;

    context.stroke(); // drawing the fixation cross
};

function gabor_mu(c) {
    // function takes the canvas as input
    //generates 2 gabors left and right to the fixation
    var context = c.getContext("2d"); //Name of the canvas
    context.font = "20px Arial";
    context.fillStyle = "#1b6601";
    context.fillText("Please respond now.",155,170);
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
    context.filter = 'contrast('+ jsPsych.currentTrial().contrast_right +')'
    context.fillRect(305,200,100,100); //co-ordinates of the right patch
    context.stroke() // drawing the two patches

    context.beginPath();
    context.strokeStyle = "#1b6601";
    context.lineWidth= 5;
    context.filter = 'contrast(1)';
    context.strokeRect(jsPsych.currentTrial().x,200,100,100);
    context.stroke();

    context.beginPath();
    context.strokeStyle = "#000000";
    context.moveTo(230, 255); context.lineTo(270, 255); //horizontal line
    context.moveTo(251, 235); context.lineTo(251, 275); //vertical line
    context.filter = 'contrast(1)'; context.lineWidth = 3;
    context.stroke(); // drawing the fixation cross
};

function gabor_mu_wait(c) {
    // function takes the canvas as input
    //generates 2 gabors left and right to the fixation
    var context = c.getContext("2d"); //Name of the canvas
    context.font = "20px Arial";
    context.fillStyle = "#FF0000";
    context.fillText("Please wait.",195,170);
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
    context.filter = 'contrast('+ jsPsych.currentTrial().contrast_right +')'
    context.fillRect(305,200,100,100); //co-ordinates of the right patch
    context.stroke() // drawing the two patches

    context.beginPath();
    context.strokeStyle = "#FF0000";
    context.lineWidth= 5;
    context.filter = 'contrast(1)';
    context.strokeRect(jsPsych.currentTrial().x,200,100,100);
    context.stroke();

    context.beginPath();
    context.strokeStyle = "#000000";
    context.moveTo(230, 255); context.lineTo(270, 255); //horizontal line
    context.moveTo(251, 235); context.lineTo(251, 275); //vertical line
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

var duration = 1000; //trial duration
var keys = ['ArrowLeft', 'ArrowRight']; // key presses allowed when the stimulus is presented to the participants
var fix_dura = 500;
var trialcounter = 0; // A counter to keep track of number of trials presented.
var prac_blocks = [1,2,3,4];
var state = [0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,1];
var state_mu = [1,1,1,1,0,0,0,1,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,1,0,0];
var s_mu = shuffle(state_mu);
var s = shuffle(state);
var counter = 1;


// key_presses for two reward states r0 and r1. r0 is when the patch on the left (key_press = arrowleft) is more rewarding and vice versa for r1.
// Two reward continengencies of 60-40 and 90-10
// This particular grouping of key_presses allows certain criteria to be maintained so that even when the array is shuffled for every participant, the criterias are not missed.
var left_70 = [['arrowleft', 'arrowleft'],['arrowleft', 'arrowright'],['arrowleft', 'arrowright'],['arrowleft', 'arrowright'],['arrowleft'],['arrowleft'],['arrowleft', 'arrowright'],['arrowleft']];
var left_90 = [['arrowleft','arrowleft'],['arrowright','arrowleft'],['arrowleft','arrowleft'],['arrowleft','arrowleft'],['arrowleft'],['arrowleft'],['arrowleft'],['arrowleft'],['arrowleft'],['arrowleft']];
var left_100 = [['arrowleft','arrowleft'],['arrowleft','arrowleft'],['arrowleft','arrowleft'],['arrowleft','arrowleft'],['arrowleft'],['arrowleft'],['arrowleft'],['arrowleft'],['arrowleft'],['arrowleft']];
var right_70 = [['arrowright', 'arrowright'],['arrowleft', 'arrowright'],['arrowleft', 'arrowright'],['arrowleft', 'arrowright'],['arrowright'],['arrowright'],['arrowleft', 'arrowright'],['arrowright']];
var right_90 = [['arrowright','arrowright'],['arrowright','arrowleft'],['arrowright','arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright']];
var right_100 = [['arrowright','arrowright'],['arrowright','arrowright'],['arrowright','arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright']];

//Randomise the order of the array for every participant
var left_70_s = shuffle(left_70);
var left_90_s = shuffle(left_90);
var right_70_s = shuffle(right_70);
var right_90_s = shuffle(right_90);
var left_100_s = shuffle(left_100);
var right_100_s = shuffle(right_100);

//Flattens the array to get rid of sub-arrays after the shuffle
var actionleft_70 = left_70_s.flat();
var actionleft_90 = left_90_s.flat();
var actionright_70 = right_70_s.flat();
var actionright_90 = right_90_s.flat();
var actionleft_100 = left_100_s.flat();
var actionright_100 = right_100_s.flat();

var index = 0 // Index of the element to serially sample from the list of the perceptual states.
var index1 = 0
var index_mu = 0
var index_mu_1 = 0
var index_missed = 0
var action_s0 = 0;
var action_s1 = 0;

var missed = []; // Empty array to which elements will be added when participants don't respond to a particular trial.
var missed_fb = []; // Empty array to which the feedback of the missed trial would be added. Will be called when the missed trial is presented again at the end of the block of trials.
var missed_con = []; // Empty array to which the contrast of the left gabor patch of the missed trial would be added.
var missed_con1 = []; // Empty array to which the contrast of the right gabor patch of the missed trial would be added.
var missed_mu = [];
var missed_mu_con = []; // Empty array to which the contrast of the left gabor patch of the missed trial would be added.
var missed_mu_con1 = []; // Empty array to which the contrast of the right gabor patch of the missed trial would be added.
var missed_block = [];
var missed_contrast = [];


// contrast difference levels for both conditions
var high_min_neg = -0.10
var high_max_neg  = 0
var high_max = 0.10
var high_min = 0
var low_min = 0.35
var low_max = 0.45
var low_max_neg = -0.35
var low_min_neg = -0.45
var avgcon = 0.5