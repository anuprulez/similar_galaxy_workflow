import socket
import sys
import re
import json
import urlparse

import predict_next_node

class ToolPredictionServer:
    """ A script to predict next tool for a sequence """

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.tools_dictionary_path = "../data/data_rev_dict.txt"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print( "Usage: python viz_server.py <port>" )
        exit( 1 )
    port = int( sys.argv[ 1 ] )
    # Create communication socket and listen on port 80.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind( ( 'localhost', port ) )
    server.listen( 5 )
    # Server loop.
    while True:
        print("Server running at: %s" % ( 'http://localhost:' + str( port ) ))
        print("\x1b[1m Waiting for requests on port %d ... \x1b[0m" % port)
        (client, address) = server.accept()
        print(client)
        print(address)
        request = client.recv( 8192 ).decode( "utf8" )
        print( "the request is " + request )
        content_type = "text/plain"
        content = ""
        match_searchpage = re.match( "^GET / HTTP/1.1", request )
        match = re.match( "^GET /(.*) HTTP/1.1", request )
        if match_searchpage:
            home_page = "create_workflow.html"
            with open( home_page ) as file:
                content = file.read()
                content_type = "text/html"
        elif match:
            query = match.group( 1 )
            next_node = predict_next_node.PredictNextNode()
            server_ob = ToolPredictionServer()
            content = ""
            parsed_query = urlparse.urlparse( query )
            params = urlparse.parse_qs( parsed_query.query )
            if( "tools_list" in query ):
                data = next_node.get_file_dictionary( server_ob.tools_dictionary_path )
                content += json.dumps(data) + '\n'
            elif( "tool_seq" in query ):
                data = next_node.find_next_nodes( params[ "tool_seq" ][ 0 ] )
                content = json.dumps( data ) + '\n'
            else:
                try:
                    # add resource files
                    with open(query) as file:
                        content = file.read()
                        if query.endswith( ".html" ):
                            content_type = "text/html"
                        elif query.endswith( ".js" ):
                            content_type = "application/javascript"
                        elif query.endswith( ".css" ):
                            content_type = "text/css"
                except:
                    content = ""
        content_length = len( content )
        answer = "HTTP/1.1 200 OK\r\n" \
            "Content-Length: %d\r\n" \
            "Content-Type: %s  \r\n" \
            "\r\n %s" % ( content_length, content_type, content )
        client.send( answer )
        client.close()
