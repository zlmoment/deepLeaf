
function addmarker(latilongi) {
    var marker = new google.maps.Marker({
        position: latilongi,
        icon: icon,
        title: 'new marker',
        draggable: true,
        map: map
    });
}

var drawMap = function() {

    var x, y;

    $('a#calculate').unbind('click').click(function() {
        $.getJSON($SCRIPT_ROOT + '/change', {}, function(data) {
            $('#dx').text(data.x);
            $('#dy').text(data.y);
            x = $('#dx').text();
            y = $('#dy').text();

            var latitude = parseFloat(x);
            var longitude = parseFloat(y);
            var point = new google.maps.LatLng(latitude, longitude);
            addmarker(point);
        });
    });

    //make data loading automatically
    var trigeerClick = function() {
        $('a#calculate').trigger('click');
    }
    setInterval(trigeerClick, Math.random()*500+100);
    

}
drawMap();
