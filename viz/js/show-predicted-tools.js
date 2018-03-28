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
            $elActualTools = $( ".actual-tools" ),
            $elAllPaths = $( ".all-included-paths" );
        $elLoading.show();
        $elPredictedTools.empty();
        $elActualTools.empty();
        $elAllPaths.empty();
        $.getJSON( predictUrl, function( data ) {
            let predictedNodes = data[ "predicted_nodes" ],
                allInputPaths = data[ "all_input_paths" ],
                predictedProb = data[ "predicted_prob" ],
                correctPredictedNodes = data[ "actual_predicted_nodes" ],
                actualNextNodes = data[ "actual_labels" ],
                presentCounter = 0,
                predictionRate = 0.0,
                toolsTemplate = "",
                correctToolsTemplate = "",
                pathsTemplate = "";
            if( Object.keys( predictedNodes ).length > 0 ) {
                predictedNodeList = predictedNodes.split( "," );
                toolsTemplate = "<ol>";
                correctToolsTemplate = "<ol>";
                for( let counter = 0; counter < 2 * actualNextNodes.length; counter++ ) {
                    let nodeName = predictedNodeList[ counter ],
                        isTrue = correctPredictedNodes[ nodeName ],
                        prob = ( predictedProb[ counter ] * 100 ).toFixed( 2 );
                    if ( isTrue || isTrue === 'true' ) {
                        toolsTemplate += "<li class='present-node'>" + nodeName + "</li>";
                        presentCounter += 1;
                    }
                    else {
                        toolsTemplate += "<li class='absent-node'>" + nodeName + "</li>";
                    }
                }
                for( let counter = 0; counter < actualNextNodes.length; counter++ ) {
                    correctToolsTemplate += "<li class='present-node'>" + actualNextNodes[ counter ] + "</li>";
                }
                toolsTemplate += "</ol>";
                correctToolsTemplate += "</ol>";
                predictionRate = ( presentCounter / actualNextNodes.length ) * 100;
                toolsTemplate += "<div>Prediction rate: <b>" + predictionRate.toFixed( 2 ) + "%</b></div>";
            }
            else {
                toolsTemplate = "No predicted tools";
            }

            if( Object.keys( allInputPaths ).length > 0 ) {
                pathsTemplate = "<ol>";
                for( var index in allInputPaths ) {
                    let toolsPath = allInputPaths[ index ],
                        toolItems = toolsPath.split( "," );
                    if( toolItems.length > 1 ) {
                        toolsPath = toolsPath.split( "," ).join( " &rarr; " );
                        pathsTemplate += "<li>" + toolsPath + "</li>";
                    }
                }
                pathsTemplate += "</ol>";
            }
            else {
                pathsTemplate = "No paths involving this sequence in the complete data"
            }
            $elLoading.hide();
            $elActualTools.append( correctToolsTemplate );
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

