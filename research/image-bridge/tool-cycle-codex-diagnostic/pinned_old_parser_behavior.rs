use ashkelon::usage::parser_for;
use ashkelon::wire::Wire;

#[test]
fn pinned_parser_misclassifies_done_item_when_completed_output_is_empty() {
    let stream = concat!(
        "event: response.output_item.done\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":2,\"item\":{\"type\":\"custom_tool_call\",\"call_id\":\"diagnostic-call-2\",\"name\":\"diagnostic-tool\",\"input\":\"{}\"}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"diagnostic-response\",\"model\":\"diagnostic-model\",\"status\":\"completed\",\"output\":[]}}\n\n",
    );
    let mut parser = parser_for(Wire::OpenAiResponses).expect("Responses parser");
    parser.feed(stream.as_bytes());
    let summary = parser.finish();

    assert!(summary.tool_calls.is_empty());
    assert!(summary.turn_end);
}
