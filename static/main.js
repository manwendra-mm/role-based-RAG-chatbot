function ask() {
    const role = document.getElementById('role').value;
    const question = document.getElementById('question').value;
    fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({role, question})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('answer').innerText = data.answer;
    });
}

function login() {
    console.log("Login function called");
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const role = document.getElementById('domain').value;
    console.log("Login attempt with", {username, password, role});
    // fetch('/login', {
    //     method: 'POST',
    //     headers: {'Content-Type': 'application/json'},
    //     body: JSON.stringify({username, password})
    // })
    // .then(response => response.json())
    // .then(data => {
    //     if (data.success) {
    //         window.location.href = '/search';
    //     } else {
    //         alert(data.message);
    //     }
    // });
}