$(document).ready(function() {
    let toolsData = null,
        toolsSeqTemplate = "",
        toolsSeq = "",
        host = document.location.host,
        toolsUrl = "http://" + host + "/tools_list=true";
    if ( toolsUrl === "" ) {
        console.error( "Error in loading text file" );
        return;
    }
    $.getJSON( toolsUrl, function( data ) {
        let toolsListOptions = "";
        toolsData = sortDictionary( data );
        for( let counter = 0; counter < toolsData.length; counter++ ) {
            toolsListOptions += "<option value='" + counter + "'>" + toolsData[ counter ] + "</options>";
        }
        $( ".tool-ids" ).append( toolsListOptions );
    });

    // Fire on change of a workflow to show all the steps for the workflow and its directed graph
    $( ".tool-ids" ).on( 'change', function( e ) {
        let selectedTool = e.target.value;
        if( toolsSeqTemplate === "" ) {
            toolsSeqTemplate = toolsData[ selectedTool ];
            toolsSeq = toolsData[ selectedTool ];
        }
        else {
            toolsSeqTemplate = " &rarr; " + toolsData[ selectedTool ];
            toolsSeq += "," + toolsData[ selectedTool ];
        }
        getPredictedTools( toolsSeq );
        $( ".workflow-so-far" ).append( toolsSeqTemplate );
        $( ".tool-ids" ).prop( 'selectedIndex', 0 );
    });

    function sortDictionary( dictionary ) {
        let toolsList = [];
        for( var key in dictionary ) {
            toolsList.push( dictionary[ key ] )
        }
        return toolsList.sort();
    }

    function getPredictedTools( toolSeq ) {
        let predictUrl =  "http://" + host + "/?predict_tool=true&tool_seq=" + toolSeq,
            $elLoading = $( ".loading-image" ),
            $elPredictedTools = $( ".predicted-tools" ),
            $elAllPaths = $( ".all-included-paths" );
        $elLoading.show();
        $elPredictedTools.empty();
        $elAllPaths.empty();
        $.getJSON( predictUrl, function( data ) {
            let predictedNodes = data[ "predicted_nodes" ],
                allInputPaths = data[ "all_input_paths" ],
                toolsTemplate = "",
                pathsTemplate = "";
            if( Object.keys( predictedNodes ).length > 0 ) {
                predictedNodeList = predictedNodes.split( "," );
                toolsTemplate = "<ul>";
                for( let counter = 0; counter < predictedNodeList.length; counter++ ) {
                    toolsTemplate += "<li>" + predictedNodeList[ counter ] + "</li>";
                }
                toolsTemplate += "</ul>";
            }
            else {
                toolsTemplate = "No predicted tools";
            }

            if( Object.keys( allInputPaths ).length > 0 ) {
                pathsTemplate = "<ul>";
                for( var index in allInputPaths ) {
                    let toolsPath = allInputPaths[ index ],
                        toolItems = toolsPath.split( "," );
                    if( toolItems.length > 1 ) {
                        toolsPath = toolsPath.split( "," ).join( " &rarr; " );
                        pathsTemplate += "<li>" + toolsPath + "</li>";
                    }
                }
                pathsTemplate += "</ul>";
            }
            else {
                pathsTemplate = "No paths involving this sequence in the complete data"
            }
            $elLoading.hide();
            $elPredictedTools.append( toolsTemplate );
            $elAllPaths.append( pathsTemplate );
        });
    }
    function clearDataEvent() {
        $( ".clear-data" ).on( "click", function( e ) {
            document.location.reload();
        });
    }
    clearDataEvent();
});

