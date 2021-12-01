var params = {
}
var arr_keys = new Array();
for (key in params) {
    arr_keys.push(key);

}
var str = "";
arr_keys.sort();
for (var i = 0; i < arr_keys.length; i++) {
    str += arr_keys[i] + params[arr_keys[i]];
}
var crypto = require("crypto").createHash("sha1");
var signature = crypto.update(str, 'utf8').digest('hex');
console.log(signature)