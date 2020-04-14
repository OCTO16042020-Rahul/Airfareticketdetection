
import urllib.parse
import urllib.request
def otpsendingfunction(mobile,nessage):
    authkey = "175606AVhvZO37X59c2613b"  # Your authentication key.
    mobiles = mobile  # Multiple mobiles numbers separated by comma.
    message = nessage  # Your message to send.
    sender = "ATTEND"  # Sender ID,While using route4 sender id should be 6 characters long.
    route = "route4"  # Define route
    # Prepare you post parameters
    values = {
        'authkey': authkey,
        'mobiles': mobiles,
        'message': message,
        'sender': sender,
        'route': route
    }
    url = "http://api.msg91.com/api/sendhttp.php"  # API URL
    postdata = urllib.parse.urlencode(values).encode("utf-8")  # URL encoding the data here.
    req = urllib.request.Request(url, postdata)
    response = urllib.request.urlopen(req)
    output = response.read()  # Get Response
    print(output)

#otpsendingfunction(9004850819)