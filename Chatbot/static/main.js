/**
 * Returns the current datetime for the message creation.
 */
function getCurrentTimestamp() {
	return new Date();
}


/**
 * Function to make the input box change the height dynamically
 */

// Function to auto resize the input field based on content
function autoResizeInput() {
    const input = document.getElementById('msg_input');
    input.style.height = 'auto'; // Reset height
    input.style.height = (input.scrollHeight) + 'px'; // Set new height based on scroll height
}

// Attach the autoResizeInput function to the input event of the message input
document.getElementById('msg_input').addEventListener('input', autoResizeInput);

/**
 * Renders a message on the chat screen based on the given arguments.
 * This is called from the `showUserMessage` and `showBotMessage`.
 */
function renderMessageToScreen(args) {
	// local variables
	let displayDate = (args.time || getCurrentTimestamp()).toLocaleString('en-IN', {
		month: 'short',
		day: 'numeric',
		hour: 'numeric',
		minute: 'numeric',
	});
	let messagesContainer = $('.messages');

	// init element
	let message = $(`
	<li class="message ${args.message_side}">
		<div class="avatar"></div>
		<div class="text_wrapper">
			<div class="text">${args.text}</div>
			<div class="timestamp">${displayDate}</div>
		</div>
	</li>
	`);

	// add to parent
	messagesContainer.append(message);

	// animations
	setTimeout(function () {
		message.addClass('appeared');
	}, 0);
	messagesContainer.animate({ scrollTop: messagesContainer.prop('scrollHeight') }, 300);
}

/* Sends a message when the 'Enter' key is pressed.
 */
$(document).ready(function() {
    $('#msg_input').keydown(function(e) {
        // Check for 'Enter' key
        if (e.key === 'Enter') {
            // Prevent default behaviour of enter key
            e.preventDefault();
			// Trigger send button click event
            $('#send_button').click();
        }
    });

	// Ensure input resizes as the user types
    $('#msg_input').on('input', function () {
        autoResizeInput();
    });
});

/**
 * Upload the image to the chatbot
 */

let selectedImageFile = null; // Store the selected image temporarily

// Trigger file input when 'add_img' is clicked
document.getElementById('add_img').addEventListener('click', function(event) {
	event.preventDefault();
	document.getElementById('file_input').click(); // Trigger file input
});

// Handle file selection
document.getElementById('file_input').addEventListener('change', function(event) {
	const file = event.target.files[0];
	if (file) {
		selectedImageFile = file; // Store the selected file
		showUserMessage("Image selected: " + file.name); // Show a message indicating image selection
	}
});

// When the send button is clicked, send both the message and image (if any)
document.getElementById('send_button').addEventListener('click', function(event) {
	const userMessage = document.getElementById('msg_input').value;
	if (!userMessage && !selectedImageFile) {
		alert("Please enter a message or select an image.");
		return;
	}

	// Prepare form data for the message and the image
	const formData = new FormData();
	formData.append('user_input', userMessage);

	if (selectedImageFile) {
		formData.append('file', selectedImageFile); // Add the image if it was selected
	}

	// Send message and image to the server
	fetch('/', {
		method: 'POST',
		body: formData
	})
	.then(response => response.text())
	.then(data => {
		showUserMessage(userMessage);
		if (selectedImageFile) {
			showUserMessage(`Image uploaded: <img src="/static/uploads/${selectedImageFile.name}" width="100px"/>`);
		}
		showBotMessage(data); // Show bot response

		// Clear input and image
		document.getElementById('msg_input').value = '';
		selectedImageFile = null;
		document.getElementById('file_input').value = ''; // Reset the file input
	})
	.catch(error => {
		console.error('Error:', error);
		showBotMessage('Sorry, there was an error.');
	});
});


/**
 * Displays the user message on the chat screen. This is the right side message.
 */
function showUserMessage(message, datetime) {
	renderMessageToScreen({
		text: message,
		time: datetime,
		message_side: 'right',
	});
}

/**
 * Displays the chatbot message on the chat screen. This is the left side message.
 */
function showBotMessage(message, datetime) {
	renderMessageToScreen({
		text: message,
		time: datetime,
		message_side: 'left',
	});
}

/**
 * Get input from user and show it on screen on button click.
 */
$('#send_button').on('click', function (e) {
    const userMessage = $('#msg_input').val();
    
    // get and show message and reset input
    showUserMessage(userMessage);
    $('#msg_input').val('');

    // send user message to FastAPI backend
    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'user_input': userMessage
        })
    })
    .then(response => response.text()) // Parse the HTML response
    .then(data => {
        // Display the bot's message
        showBotMessage(data);
    })
    .catch(error => {
        console.error('Error:', error);
        showBotMessage('Sorry, there was an error.');
    });
});

let chatHistory = []; // Store chat history

/**
 * Save conversation to history.
 */
function saveConversationToHistory(userMessage, botMessage) {
	chatHistory.push({
		user: userMessage,
		bot: botMessage
	});
	updateHistoryPanel();
}

/**
 * Update the history panel with stored conversation history.
 */
function updateHistoryPanel() {
	const historyPanel = document.getElementById('accordion'); // The panel where the history will be shown
	historyPanel.innerHTML = ''; // Clear the existing history

	chatHistory.forEach((entry, index) => {
		const panelItem = `
			<div class="panel panel-default">
				<div class="panel-heading">
					<h4 class="panel-title">
						<a class="accordion-toggle" data-toggle="collapse" data-parent="#accordion"
							href="#collapse${index}">${entry.user}</a>
					</h4>
				</div>
				<div id="collapse${index}" class="panel-collapse collapse ${index === 0 ? 'in' : ''}">
					<div class="panel-body">
						${entry.bot}
					</div>
				</div>
			</div>
		`;

		historyPanel.innerHTML += panelItem; // Append each entry to the history panel
	});
}

/**
 * Displays the user message on the chat screen. This is the right side message.
 */
function showUserMessage(message, datetime) {
	renderMessageToScreen({
		text: message,
		time: datetime,
		message_side: 'right',
	});
}

/**
 * Displays the chatbot message on the chat screen. This is the left side message.
 */
function showBotMessage(message, datetime) {
	renderMessageToScreen({
		text: message,
		time: datetime,
		message_side: 'left',
	});
}

/**
 * Handles the send button click and saves conversation.
 */
document.getElementById('send_button').addEventListener('click', function(event) {
	const userMessage = document.getElementById('msg_input').value;
	if (!userMessage && !selectedImageFile) {
		alert("Please enter a message or select an image.");
		return;
	}

	// Prepare form data for the message and the image
	const formData = new FormData();
	formData.append('user_input', userMessage);

	if (selectedImageFile) {
		formData.append('file', selectedImageFile); // Add the image if it was selected
	}

	// Send message and image to the server
	fetch('/', {
		method: 'POST',
		body: formData
	})
	.then(response => response.text())
	.then(data => {
		showUserMessage(userMessage);
		if (selectedImageFile) {
			showUserMessage(`Image uploaded: <img src="/static/uploads/${selectedImageFile.name}" width="100px"/>`);
		}
		showBotMessage(data); // Show bot response

		// Save the conversation to the history
		saveConversationToHistory(userMessage, data);

		// Clear input and image
		document.getElementById('msg_input').value = '';
		selectedImageFile = null;
		document.getElementById('file_input').value = ''; // Reset the file input
	})
	.catch(error => {
		console.error('Error:', error);
		showBotMessage('Sorry, there was an error.');
	});
});

/**
 * Set initial bot message to the screen for the user.
 */
$(window).on('load', function () {
	showBotMessage('Hello there! I am your reliable fashion consultant. You can ask me about anything!');
});
