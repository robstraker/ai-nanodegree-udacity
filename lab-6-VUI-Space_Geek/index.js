'use strict';
var Alexa = require('alexa-sdk');

//=========================================================================================================================================
//TODO: The items below this comment need your attention.
//=========================================================================================================================================

//Replace with your app ID (OPTIONAL).  You can find this value at the top of your skill's page on http://developer.amazon.com.  
//Make sure to enclose your value in quotes, like this: var APP_ID = "amzn1.ask.skill.bb4045e6-b3e8-4133-b650-72923c5980f1";
var APP_ID = undefined;

var SKILL_NAME = "Space Facts";
var GET_FACT_MESSAGE = "Here's your fact: ";
var HELP_MESSAGE = "You can say tell me a space fact, or, you can say exit... What can I help you with?";
var HELP_REPROMPT = "What can I help you with?";
var STOP_MESSAGE = "Goodbye!";

//=========================================================================================================================================
//TODO: Replace this data with your own.  You can find translations of this data at http://github.com/alexa/skill-sample-node-js-fact/data
//=========================================================================================================================================
var data = [
    "A year on Mercury is just 88 days long.",
    "Despite being farther from the Sun, Venus experiences higher temperatures than Mercury.",
    "Venus rotates counter-clockwise, possibly because of a collision in the past with an asteroid.",
    "On Mars, the Sun appears about half the size as it does on Earth.",
    "Earth is the only planet not named after a god.",
    "Jupiter has the shortest day of all the planets.",
    "The Milky Way galaxy will collide with the Andromeda Galaxy in about 5 billion years.",
    "The Sun contains 99.86% of the mass in the Solar System.",
    "The Sun is an almost perfect sphere.",
    "A total solar eclipse can happen once every 1 to 2 years. This makes them a rare event.",
    "Saturn radiates two and a half times more energy into space than it receives from the sun.",
    "The temperature inside the Sun can reach 15 million degrees Celsius.",
    "The Moon is moving approximately 3.8 cm away from our planet every year.",
    "Mercury and Venus are the only two planets in our solar system that do not have any moons."
	"If a star passes too close to a black hole, it can be torn apart.",
	"The hottest planet in our solar system is Venus. Most people often think that it would be Mercury, as it's the closest planet to the sun. This is because Venus has a lot of gasses in its atmosphere, which causes the Greenhouse Effect.",
	"The solar system is around 4.6 billion years old. Scientist estimate that it will probably last another 5000 million years",
	"Enceladus, one of Saturn&#8217;s smaller moons, reflects some 90% of the sunlight, making it more reflective than snow!",
	"The highest mountain known to man is the Olympus Mons, which is located on Mars. It's peak is 15 miles (25KM) high, making it nearly 3 times higher than Mt Everest.",
	"The Whirlpool Galaxy (M51) was the very first celestial object to be identified as being spiral.",
	"A light year is the distance covered by light in a single year, this is equivalent to 5.88 trillion miles (9.5 trillion KM)!",
	"The width of the Milky Way is around 100,000 light years.",
	"The Sun is over 300.000 times larger than the Earth.",
	"Footprints and tire tracks left by astronauts on the moon will stay there forever as there is no wind to blow them away.",
	"Because of lower gravity, a person who weighs 100kg on earth would only weigh 38kg on the surface of Mars.",
	"Scientists believe there are 67 moons that orbit Jupiter, however only 53 of these have been named.",
	"The Martian day is 24 hours 39 minutes and 35 seconds.",
	"NASA's Crater Observation and Sensing Satellite (LCROSS) declared that they have found evidence of significant amounts of water on the Earth's Moon.",
	"The Sun makes a full rotation once every 25-35 days.",
	"Venus is the only planet that spins backwards relative to the other planets.",
	"The force of gravity can sometimes cause comets to tear apart.",
	"It is thanks to the Sun and our own moons gravity that we have high and low tides.",
	"Pluto is smaller than the Earth&#8217;s moon!",
	"According to mathematics, white holes are possible, although as of yet, we have found none.",
	"Our moon is around 4.5 billion years old.",
	"There are more volcanoes on Venus than any other planet within our solar system.",
	"Uranus' blue glow is down to the methane in its atmosphere, which filters out all the red light.",
	"The four planets in our solar system that are known as gas giants are Jupiter, Neptune, Saturn and Uranus."
];

//=========================================================================================================================================
//Editing anything below this line might break your skill.  
//=========================================================================================================================================
exports.handler = function(event, context, callback) {
    var alexa = Alexa.handler(event, context);
    alexa.APP_ID = APP_ID;
    alexa.registerHandlers(handlers);
    alexa.execute();
};

var handlers = {
    'LaunchRequest': function () {
        this.emit('GetNewFactIntent');
    },
    'GetNewFactIntent': function () {
        var factArr = data;
        var factIndex = Math.floor(Math.random() * factArr.length);
        var randomFact = factArr[factIndex];
        var speechOutput = GET_FACT_MESSAGE + randomFact;
        this.emit(':tellWithCard', speechOutput, SKILL_NAME, randomFact)
    },
    'AMAZON.HelpIntent': function () {
        var speechOutput = HELP_MESSAGE;
        var reprompt = HELP_REPROMPT;
        this.emit(':ask', speechOutput, reprompt);
    },
    'AMAZON.CancelIntent': function () {
        this.emit(':tell', STOP_MESSAGE);
    },
    'AMAZON.StopIntent': function () {
        this.emit(':tell', STOP_MESSAGE);
    }
};