from langchain.prompts import PromptTemplate


create_graph_prompt = PromptTemplate.from_template(
    "You have an example of dialogue from customer chatbot system. You also have an "
    "example of set of rules how chatbot system works should be looking - it is "
    "a set of nodes when chatbot system respons and a set of transitions that are "
    "triggered by user requests. "
    "Here is the example of set of rules: "
    "'edges': [ [ 'source': 1, 'target': 2, 'utterances': 'I need to make an order' ], "
    "[ 'source': 1, 'target': 2, 'utterances': 'I want to order from you' ], "
    "[ 'source': 2, 'target': 3, 'utterances': 'I would like to purchase 'Pale Fire' and 'Anna Karenina', please' ], "
    "'nodes': [ [ 'id': 1, 'label': 'start', 'is_start': true, 'utterances': [ 'How can I help?', 'Hello' ], "
    "[ 'id': 2, 'label': 'ask_books', 'is_start': false, 'utterances': [ 'What books do you like?'] ] "
    "I will give a dialogue, your task is to build a graph for this dialogue in the format above. We allow several edges with equal "
    "source and target and also multiple responses on one node so try not to add new nodes if it is logical just to extend an "
    "exsiting one. utterances in one node or on multiedge should close between each other and correspond to different answers "
    "to one question or different ways to say something. For example, for question about preferences or a Yes/No question "
    "both answers can be fit in one multiedge, there’s no need to make a new node. If two nodes has the same responses they "
    "should be united in one node. Do not make up utterances that aren’t present in the dialogue. Please do not combine "
    "utterances for multiedges in one list, write them separately like in example above. Every utterance from the dialogue, "
    "whether it is from user or assistanst, should contain in one of the nodes. Edges must be utterances from the user. Do not forget ending nodes with goodbyes. "
    "Sometimes dialogue can correspond to several iterations of loop, for example: "
    "['text': 'Do you have apples?', 'participant': 'user'], "
    "['text': 'Yes, add it to your cart?', 'participant': 'assistant'], "
    "['text': 'No', 'participant': 'user'], "
    "['text': 'Okay. Anything else?', 'participant': 'assistant'], "
    "['text': 'I need a pack of chips', 'participant': 'user'], "
    "['text': 'Yes, add it to your cart?', 'participant': 'assistant'], "
    "['text': 'Yes', 'participant': 'user'], "
    "['text': 'Done. Anything else?', 'participant': 'assistant'], "
    "['text': 'No, that’s all', 'participant': 'user'], "
    "it corresponds to following graph: "
    "[ nodes: "
    "'id': 1, "
    "'label': 'confirm_availability_and_ask_to_add', "
    "'is_start': false, "
    "'utterances': 'Yes, add it to your cart?' "
    "], "
    "[ "
    "'id': 2, "
    "'label': 'reply_to_yes', "
    "'is_start': false, "
    "'utterances': ['Done. Anything else?', 'Okay. Anything else?'] "
    "], "
    "[ "
    "'id': 3, "
    "'label': 'finish_filling_cart', "
    "'is_start': false, "
    "'utterances': 'Okay, everything is done, you can go to cart and finish the order.' "
    "], "
    "edges: "
    "[ "
    "'source': 1, "
    "'target': 2, "
    "'utterances': 'Yes' "
    "], "
    "[ "
    "'source': 1, "
    "'target': 2, "
    "'utterances': 'No' "
    "], "
    "[ "
    "'source': 2, "
    "'target': 1, "
    "'utterances': 'I need a pack of chips' "
    "], "
    "[ "
    "'source': 2, "
    "'target': 3, "
    "'utterances': 'No, that’s all' "
    "]. "
    "We encourage you to use cycles and complex interwining structure of the graph with 'Yes'/'No' edges for the branching."
    "This is the end of the example. Brackets must be changed back into curly braces to create a valid JSON string. Return ONLY JSON string in plain text (no code blocks) without any additional commentaries."
    "Dialogue: {dialog}"
)

check_graph_utterances_prompt = PromptTemplate.from_template(
    "You have a dialogue and a structure of graph built on this dialogue it is a "
    "set of nodes when chatbot system responses and a set of transitions that are triggered by user requests.\n"
    "Please say if for every utterance in the dialogue there exist either a utteranse in node or in some edge. "
    "be attentive to what nodes we actually have in the 'nodes' list, because some nodes from the list of edges maybe non existent:\n"
    "Graph: {graph}\n"
    "Dialogue: {dialog}\n"
    "just print the list of utteance and whether there exsit a valid edge of node contating it, if contains print the node or edge"
)

check_graph_validity_prompt = PromptTemplate.from_template(
    "1. You have an example of dialogue from customer chatbot system.\n"
    "2. You also have a set of rules how chatbot system works - a set of "
    "nodes when chatbot system respons and a set of transitions that are triggered by user requests.\n"
    "3. Chatbot system can move only along transitions listed in 2.  If a transition from node A to "
    "node B is not listed we cannot move along it.\n"
    "4. If a dialog doesn't contradcit with the rules listed in 2 print YES otherwise if such dialog "
    "could'nt happen because it contradicts the rules print NO.\nDialogue: {dialog}.\nSet of rules: {rules}"
)

cycle_graph_generation_prompt_basic = PromptTemplate.from_template("""
    Create a complex dialogue graph where the conversation MUST return to an existing node.

    **CRITICAL: Response Specificity**
    Responses must acknowledge and build upon what the user has already specified:

    INCORRECT flow:
    - User: "I'd like to order a coffee"
    - Staff: "What would you like to order?" (TOO GENERAL - ignores that they specified coffee)

    CORRECT flow:
    - User: "I'd like to order a coffee"
    - Staff: "What kind of coffee would you like?" (GOOD - acknowledges they want coffee)

    Example of a CORRECT cyclic graph for a coffee shop:
    "edges": [
        {{ "source": 1, "target": 2, "utterances": ["Hi, I'd like to order a coffee"] }},
        {{ "source": 2, "target": 3, "utterances": ["A large latte please"] }},
        {{ "source": 3, "target": 4, "utterances": ["Yes, that's correct"] }},
        {{ "source": 4, "target": 5, "utterances": ["Here's my payment"] }},
        {{ "source": 5, "target": 2, "utterances": ["I'd like to order another coffee"] }}
    ],
    "nodes": [
        {{ "id": 1, "label": "welcome", "is_start": true, "utterances": ["Welcome! How can I help you today?"] }},
        {{ "id": 2, "label": "ask_coffee_type", "is_start": false, "utterances": ["What kind of coffee would you like?"] }},
        {{ "id": 3, "label": "confirm", "is_start": false, "utterances": ["That's a large latte. Is this correct?"] }},
        {{ "id": 4, "label": "payment", "is_start": false, "utterances": ["Great! That'll be $5. Please proceed with payment."] }},
        {{ "id": 5, "label": "completed", "is_start": false, "utterances": ["Thank you! Would you like another coffee?"] }}
    ]

    **Rules:**
    1) Responses must acknowledge what the user has already specified
    2) The final node MUST connect back to an existing node
    3) Each node must have clear purpose
    4) Return ONLY the JSON without commentary
    5) Graph must be cyclic - no dead ends
    6) All edges must connect to existing nodes
    7) The cycle point should make logical sense

    **Your task is to create a cyclic dialogue graph about the following topic:** {topic}.
    """)

cycle_graph_generation_prompt_enhanced = PromptTemplate.from_template(
    """
Create a dialogue graph for a {topic} conversation that will be used for training data generation. The graph must follow these requirements:

1. Dialogue Flow Requirements:
   - Each assistant message (node) must be a precise question or statement that expects a specific type of response
   - Each user message (edge) must logically and directly respond to the previous assistant message
   - All paths must maintain clear context and natural conversation flow
   - Avoid any ambiguous or overly generic responses

2. Graph Structure Requirements:
   - Must contain at least 2 distinct cycles (return paths)
   - Each cycle should allow users to:
     * Return to previous choices for modification
     * Restart specific parts of the conversation
     * Change their mind about earlier decisions
   - Include clear exit points from each major decision path
   
3. Core Path Types:
   - Main success path (completing the intended task)
   - Multiple modification paths (returning to change choices)
   - Early exit paths (user decides to stop)
   - Alternative success paths (achieving goal differently)

Example of a good cycle structure:
Assistant: "What size coffee would you like?"
User: "Medium please"
Assistant: "Would you like that hot or iced?"
User: "Actually, can I change my size?"
Assistant: "Of course! What size would you like instead?"

Format:
{{
    "edges": [
        {{
            "source": "node_id",
            "target": "node_id",
            "utterances": ["User response text"]
        }}
    ],
    "nodes": [
        {{
            "id": "node_id",
            "label": "semantic_label",
            "is_start": boolean,
            "utterances": ["Assistant message text"]
        }}
    ]
}}

Requirements for node IDs:
- Must be unique integers
- Start node should have ID 1
- IDs should increment sequentially

Return ONLY the valid JSON without any additional text or explanations.
"""
)

cycle_graph_repair_prompt = PromptTemplate.from_template("""
Fix the invalid transitions in this dialogue graph while keeping its structure.

Current invalid transitions that need to be fixed:
{invalid_transitions}

Original graph structure:
{graph_json}

Requirements for the fix:
1. Keep all node IDs and structure the same
2. Fix ONLY the invalid transitions
3. Make sure the fixed transitions are logical and natural
4. Each user response must logically follow from the assistant's previous message
5. Each assistant response must properly address the user's input

Return ONLY the complete fixed graph JSON with the same structure.
""")
