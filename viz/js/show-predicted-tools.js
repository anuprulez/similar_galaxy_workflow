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
            toolsSeqTemplate = " > " + toolsData[ selectedTool ];
            toolsSeq += "," + toolsData[ selectedTool ];
        }
        getPredictedTools( toolsSeq );
        $( ".workflow-so-far" ).append( toolsSeqTemplate );
    });

    function sortDictionary( dictionary ) {
        let toolsList = [];
        for( var key in dictionary ) {
            toolsList.push( dictionary[ key ] )
        }
        return toolsList.sort();
    }

    function getPredictedTools( toolSeq ) {
        let predictUrl =  "http://" + host + "/?predict_tool=true&tool_seq=" + toolSeq;
        $.getJSON( predictUrl, function( data ) {
            let predictedNodes = data[ "predicted_nodes" ],
                toolsTemplate = "";
            if( Object.keys( predictedNodes ).length > 0 ) {
                predictedNodeList = predictedNodes.split( "," );
                toolsTemplate = "<ul>";
                for( let counter = 0; counter < predictedNodeList.length; counter++ ) {
                    toolsTemplate += "<li>" + predictedNodeList[ counter ] + "</li>";
                }
                toolsTemplate += "</ul>";
            }
            else {
                toolsTemplate = "No predicted tools"
            }
            $( ".predicted-tools" ).empty().append( toolsTemplate );
        });
    }
});

