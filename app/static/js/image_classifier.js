function getResults(error, results){
    if (error) {
        console.error(error)
    } else {

        // Remove previous divs!
        var list = document.getElementById("result");
        while (list.hasChildNodes()) {
            list.removeChild(list.firstChild);
        }

        var div = document.createElement("DIV");

        results.forEach(function(entry) {
            // console.log(entry.label);
            // console.log(entry.confidence);
            // Create a new div
            var result = document.createElement("DIV");

            const confidence = entry.confidence;
            switch (true) {
                case (confidence < 0.10):
                    result.className = "lo";
                    result.innerHTML =  entry.label + " 🤖: " + entry.confidence ;
                    break;
                case (confidence < 0.50):
                    result.className = "mi";
                    result.innerHTML = entry.label + " 🙌: " + entry.confidence ;
                    break;
                case (confidence >= 0.50):
                    result.className = "hi";
                    result.innerHTML = entry.label + " 🧠: " + entry.confidence ;
                    break;
                default:
                    result.className = "res";
                    result.innerHTML = "🧠" + entry.label + ": " + entry.confidence ;
                    break;
            }

            result.id = "res";

            // result.innerHTML = entry.label + ": " + entry.confidence ;         // Create a text node
            document.getElementById("result").appendChild(result);
        });

        $('#result').fadeIn(200);

        $('.input_file').show();
        $('.loaded_image').show();
        $('.loader').hide();
    }
}


async function readURL(input) {
    // Show loading animation
    $('.input_file').hide();
    $('.loaded_image').hide();
    $('.result').hide()
    $('.loader').show();

    if (input.files && input.files[0]) {

        var reader = new FileReader();
        var currFile = input.files[0];

        reader.onload = (function(theFile){
            var fileName = theFile.name;
            return async function(event){
                dataUri = event.target.result

                $('#loaded_image').attr('src', dataUri);

                img = new Image();
                img.src = dataUri;

                classifier = await ml5.imageClassifier('MobileNet');
                classifier.classify(img, getResults);
            };
        })(currFile);

        reader.readAsDataURL(currFile);
    }
}