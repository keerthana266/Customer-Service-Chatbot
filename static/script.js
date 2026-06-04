window.onload = () => {

    const history = localStorage.getItem("chatHistory");

    if (history) {
        document.getElementById("chatbox").innerHTML = history;
    }

    const chatbox = document.getElementById("chatbox");
    chatbox.scrollTop = chatbox.scrollHeight;
};

async function sendMessage() {

    const messageInput = document.getElementById("message");
    const message = messageInput.value.trim();

    if (message === "") {
        return;
    }

    const chatbox = document.getElementById("chatbox");

    const currentTime = new Date().toLocaleTimeString();

    // User Message
    chatbox.innerHTML += `
        <div class="user">
            👤 ${message}
            <small>${currentTime}</small>
        </div>
    `;

    chatbox.scrollTop = chatbox.scrollHeight;

    try {

        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        const data = await response.json();

        chatbox.innerHTML += `
            <div class="bot">
                🤖 ${data.message}
                <small>${currentTime}</small>
                <span class="confidence">
                    Confidence: ${data.confidence}%
                </span>
            </div>
        `;

        // Save chat history
        localStorage.setItem(
            "chatHistory",
            chatbox.innerHTML
        );

        chatbox.scrollTop = chatbox.scrollHeight;

    } catch (error) {

        chatbox.innerHTML += `
            <div class="bot">
                🤖 Error connecting to server.
            </div>
        `;
    }

    messageInput.value = "";
}

document.getElementById("message").addEventListener("keypress", function(event) {

    if (event.key === "Enter") {
        sendMessage();
    }

});