var timeline = []; //array with every stage of the experiment.
var duration = 1000; //trial duration
var keys = ['arrowleft', 'arrowright']; // key presses allowed when the stimulus is presented to the participants
var fix_dura = 500;
var trialcounter = 0; // A counter to keep track of number of trials presented.
var prac_blocks = [1,2,3,4];
var state = [0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,1];
var s = shuffle(state);
var counter = 1;

// key_presses for two reward states r0 and r1. r0 is when the patch on the left (key_press = arrowleft) is more rewarding and vice versa for r1.
// Two reward continengencies of 60-40 and 90-10
// This particular grouping of key_presses allows certain criteria to be maintained so that even when the array is shuffled for every participant, the criterias are not missed.
var left_60 = [['arrowleft', 'arrowright'],['arrowleft', 'arrowright'],['arrowleft', 'arrowright'],['arrowleft', 'arrowright'],['arrowleft'],['arrowleft'],['arrowleft', 'arrowright'],['arrowleft']];
var left_90 = [['arrowleft','arrowleft'],['arrowright','arrowleft'],['arrowleft','arrowleft'],['arrowleft','arrowleft'],['arrowleft'],['arrowleft'],['arrowleft'],['arrowleft'],['arrowleft'],['arrowleft']];
var right_60 = [['arrowleft', 'arrowright'],['arrowleft', 'arrowright'],['arrowleft', 'arrowright'],['arrowleft', 'arrowright'],['arrowright'],['arrowright'],['arrowleft', 'arrowright'],['arrowright']];
var right_90 = [['arrowright','arrowright'],['arrowright','arrowleft'],['arrowright','arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright'],['arrowright']];

//Randomise the order of the array for every participant
var left_60_s = shuffle(left_60);
var left_90_s = shuffle(left_90);
var right_60_s = shuffle(right_60);
var right_90_s = shuffle(right_90);

//Flattens the array to get rid of sub-arrays after the shuffle
var actionleft_60 = left_60_s.flat();
var actionleft_90 = left_90_s.flat();
var actionright_60 = right_60_s.flat();
var actionright_90 = right_90_s.flat();

var index = 0 // Index of the element to serially sample from the list of the perceptual states.
var index1 = 0
var action_s0 = 0;
var action_s1 = 0;

var missed = []; // Empty array to which elements will be added when participants don't respond to a particular trial.
var missed_fb = []; // Empty array to which the feedback of the missed trial would be added. Will be called when the missed trial is presented again at the end of the block of trials.
var missed_con = []; // Empty array to which the contrast of the left gabor patch of the missed trial would be added.
var missed_con1 = []; // Empty array to which the contrast of the right gabor patch of the missed trial would be added.
var prac_blocks = [1,2,3,4]; // block counter for practice blocks.

// contrast difference levels for both conditions
var high_min_neg = -0.08
var high_min = 0
var high_max = 0.08
var low_min = 0.3
var low_max = 0.38
var low_max_neg = -0.3
var low_min_neg = -0.38
var avgcon = 0.5
