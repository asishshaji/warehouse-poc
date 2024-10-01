class AgentAction:
    def __init__(self, action_type, params):
        self.action_type = action_type
        self.params = params


class Orchestrator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.context = []
        self.rags = {}  # maintains a dictionary to the rags

    def perceive(self, query):
        # query the docs collection to get the document context
        document_context = self.get_context_from_documents(query)
        self.context.append(f"Given the following context: {document_context}")
        self.context.append("")

    def decide(self):
        prompt = f"""
        Given the following context
        {" ".join(self.context)}
        Decide on the actions to take. Options are
        1. generate_sql_query
        2. propose_solution
        3. request_more_info
        """

    def act(self, action: AgentAction):
        pass

    def get_context_from_documents(self, query):
        pass


if __name__ == "__main__":
    orch = Orchestrator(model_name="")
    query = "Why is my epson printer not working"
    orch.perceive(query)

    action = orch.decide()
    orch.act(action=action)
