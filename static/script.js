document.getElementById("query-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    // Clear previous results
    document.getElementById("progress-text").innerText = "Processing query...";
    document.getElementById("user-question").innerText = "N/A";
    document.getElementById("answer-box").innerText = "N/A";

    // Get user inputs
    const role = document.getElementById("role").value;
    const question = document.getElementById("question").value;

    // Prepare data
    const data = { role, question };

    try {
        // Fetch results from backend
        const response = await fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        // Update progress and results
        if (result.answer) {
            document.getElementById("progress-text").innerText = "Query completed!";
            document.getElementById("user-question").innerText = result.question;
            document.getElementById("answer-box").innerText = result.answer;
        } else if (result.error) {
            document.getElementById("progress-text").innerText = `Error: ${result.error}`;
        }
    } catch (error) {
        document.getElementById("progress-text").innerText = "An error occurred while processing your query.";
    }
});