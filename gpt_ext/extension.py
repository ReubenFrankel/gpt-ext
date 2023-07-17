"""Meltano OpenAI extension."""
from __future__ import annotations

import json
import os
import sys
import csv
import pprint
from typing import Any

import typer

from gpt_ext.ai import get_chain, load_chroma_vectorstore
from gpt_ext.edk_fixes.extension_base import CLI, ExtensionBase

DEFAULT_SETTING_VALS = {
    "chroma_dir": os.path.join(os.path.expanduser("~"), ".chroma"),
}

class GPTExt(ExtensionBase):
    """Extension implementing the ExtensionBase interface."""

    supports_invoke_command = False
    supports_init_command = True

    def __init__(self) -> None:
        """Initialize the extension."""
        # self.openai_bin = "OpenAI"  # verify this is the correct name
        # self.openai_invoker = Invoker(self.openai_bin)
        self._vectorstore: Any | None = None

    @property
    def vectorstore(self) -> Any:
        if not self._vectorstore:
            _vectorstore = load_chroma_vectorstore(self.get_config("chroma_dir"))

        return _vectorstore

    def get_config(self, setting_name: str) -> Any:
        """Get a config setting."""
        env_var = "OPENAI_" + setting_name.upper()
        if env_var in os.environ:
            return os.environ[env_var]

        return DEFAULT_SETTING_VALS.get(setting_name, None)

    @CLI.command()
    @staticmethod
    def chat(
        ctx: typer.Context,
        questions: str = typer.Option(..., prompt="What is your question?"),
    ) -> None:
        """Invoke the plugin.

        Note: that if a command argument is a list, such as command_args,
        then unknown options are also included in the list and NOT stored in the
        context as usual.
        """
        app: OpenAI = ctx.obj

        def question_handler(text):
            print("Question:", text)
            result = typer.prompt("???")
            return result

        def stream_handler(text):
            print("Stream:", text, file=sys.stderr)
       

        qa = get_chain(
            app.vectorstore,
            question_handler=question_handler,
            stream_handler=stream_handler,
        )
            
        answers = []
        chat_history = []
        if len(questions) > 0:
            for question in questions.split(","):
                answers.append(question)

                if "?" in question:
                    #result = qa({"question": question})
                    result = qa({"question": question, "chat_history": chat_history})
                    # pprint.pprint(result)
                    docs_and_similarities = (
                        app.vectorstore.similarity_search_with_relevance_scores(question)
                    )
                    context = [{
                        "page_content": doc.page_content,
                        "similarity": similarity,
                    } for doc, similarity in docs_and_similarities]

                    chat_history.append((question, result["answer"]))
                    answers.append(result["answer"].strip())
                    answers.append(json.dumps(context))

        csv_writer = csv.writer(sys.stdout)
#        csv_writer.writerow(questions)
        csv_writer.writerow(answers)

        # csv_string = ",".join(str(i) for i in answers)
        # print(csv_string)
