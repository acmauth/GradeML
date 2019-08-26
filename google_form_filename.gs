function getFileNames() {
  var sheet = SpreadsheetApp.getActive()
  var links = sheet.getRange("M2:M").getValues();
  var filenames = []; 
  for (var i = 0; i < links.length; i++) {
    var url = links[i][0];
    if (url != "") {
      var params = url.split("?")[1]
      var id = params.split("=")[1];
      var filename = DriveApp.getFileById(id).getName();
      filenames.push([filename]);
    }
  }
  var fileLength = filenames.length + 1
  var fileNameColumn = "P"; // Column P
  var destination = sheet.getRange(fileNameColumn + "2" + ":" + fileNameColumn + fileLength);
  destination.setValues(filenames);
}