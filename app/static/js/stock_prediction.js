
function prediction(input) {
    console.log(input.value)
    // Show loading animation
    $('.days').hide();
    $('.input_days').hide();
    $('.prediction').hide();
    $('.tweets').hide();
    $('.loader').show();
    if (input.value > 0) {
        data = "days="+input.value

        console.log(data)
        var xhr = new XMLHttpRequest();
        xhr.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                // console.log(this.response);
                var parsed_data = JSON.parse(this.response);
                $("#prediction").html(parsed_data.result.prediction);
                $("#confidence").html(parsed_data.result.confidence);
                // $("#result").html(parsed_data.confidence);

                $('.days').show();
                $('.input_days').show();
                $('.tweets').show();
                $('.prediction').show();
                // $('.sentiment_image').style("display: block;")
                $('.loader').hide();
                document.getElementById("sentiment_image").style.display = "block";
            }
        };
        xhr.open('POST', '/predict', true);
        xhr.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
        xhr.send(data);
    }
}


$(function() {
    $('button').click(function() {
        $.ajax({
            url: '/predict',
            data: $('form').serialize(),
            type: 'POST',
            success: function(response) {
                console.log(response);
                var parsed_data = JSON.parse(response);
                $("#result").html(parsed_data.days);
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});


//
//
// $(function() {
//     $('button').click(function() {
//
//         var clock={
//           "key":21 //say
//         };
//
//         data: $('form').serialize()
//
//         $.ajax({
//             url: 'predict',
//             type: 'POST',
//             dataType: 'json',
//             data: JSON.stringify(clock),
//             contentType:"application/json; charset=UTF-8"
//         })
//         .done(function(data) {
//             // do stuff here
//             console.log(data);
//         })
//         .fail(function(err) {
//             // do stuff here
//         })
//         .always(function(info) {
//             // do stuff here
//         });
//     });
// });
//
//
//
//
//
//
// function getResults(error, results){
//     if (error) {
//         console.error(error)
//     } else {
//
//         // Remove previous divs!
//         var list = document.getElementById("result");
//         while (list.hasChildNodes()) {
//             list.removeChild(list.firstChild);
//         }
//
//         var div = document.createElement("DIV");
//
//         results.forEach(function(entry) {
//             // console.log(entry.label);
//             // console.log(entry.confidence);
//             // Create a new div
//             var result = document.createElement("DIV");
//
//             const confidence = entry.confidence;
//             switch (true) {
//                 case (confidence < 0.10):
//                     result.className = "lo";
//                     result.innerHTML =  entry.label + " 🤖: " + entry.confidence ;
//                     break;
//                 case (confidence < 0.50):
//                     result.className = "mi";
//                     result.innerHTML = entry.label + " 🙌: " + entry.confidence ;
//                     break;
//                 case (confidence >= 0.50):
//                     result.className = "hi";
//                     result.innerHTML = entry.label + " 🧠: " + entry.confidence ;
//                     break;
//                 default:
//                     result.className = "res";
//                     result.innerHTML = "🧠" + entry.label + ": " + entry.confidence ;
//                     break;
//             }
//
//             result.id = "res";
//
//             // result.innerHTML = entry.label + ": " + entry.confidence ;         // Create a text node
//             document.getElementById("result").appendChild(result);
//         });
//
//         $('#result').fadeIn(200);
//
//         $('.input_file').show();
//         $('.loaded_image').show();
//         $('.loader').hide();
//     }
// }
//
//
// async function readURL(input) {
//     // Show loading animation
//     $('.input_file').hide();
//     $('.loaded_image').hide();
//     $('.result').hide()
//     $('.loader').show();
//
//     if (input.files && input.files[0]) {
//
//         var reader = new FileReader();
//         var currFile = input.files[0];
//
//         reader.onload = (function(theFile){
//             var fileName = theFile.name;
//             return async function(event){
//                 dataUri = event.target.result
//
//                 $('#loaded_image').attr('src', dataUri);
//
//                 img = new Image();
//                 img.src = dataUri;
//
//                 classifier = await ml5.imageClassifier('MobileNet');
//                 classifier.classify(img, getResults);
//             };
//         })(currFile);
//
//         reader.readAsDataURL(currFile);
//     }
// }