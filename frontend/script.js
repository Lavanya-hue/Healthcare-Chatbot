const API_URL = "http://127.0.0.1:8000/ask";

async function sendMessage() {
    let input = document.getElementById("user-input");
    let message = input.value.trim();

    if (!message) return;

    appendMessage(message, "user");
    input.value = "";

    try {
        let response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: message,
                max_new_tokens: 150
            }),
        });

        if (!response.ok) {
            appendMessage("⚠ Server Error (" + response.status + ")", "bot");
            return;
        }

        let data = await response.json();
        appendMessage(data.answer, "bot");

    } catch (error) {
        appendMessage("⚠ Could not connect to server", "bot");
    }
}

function appendMessage(text, sender) {
    let chatBox = document.getElementById("chat-box");
    let msg = document.createElement("div");
    msg.classList.add("message", sender);
    msg.textContent = text;

    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight; 
}
