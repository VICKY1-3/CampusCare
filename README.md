# CampusCare — College Complaint Management System

CampusCare is a web-based College Complaint Management System developed using Python Flask and PostgreSQL. It provides a centralized platform for students to submit complaints and administrators to review, manage, and track their resolution.

## Features

### Student Portal
- Student registration and login
- Submit complaints with a title, description, and category
- View submitted complaints and their current status
- View administrator remarks and complaint updates

### Admin Portal
- Secure administrator login
- View and manage student complaints
- Update complaint statuses: Submitted, In Progress, Resolved, and Rejected
- Add remarks when updating complaints
- View complaint statistics and analytics

### Analytics Dashboard
- Total complaint count
- Resolved and pending complaint statistics
- Complaint distribution by status
- Complaint distribution by category
- Monthly complaint submission trends
- Resolution rate

## Technologies Used

- **Backend:** Python, Flask
- **Database:** PostgreSQL
- **Frontend:** HTML, CSS, JavaScript
- **Charts:** Chart.js
- **Database Driver:** Psycopg
- **Authentication:** Flask sessions and Werkzeug password hashing
- **Development Environment:** Visual Studio Code

## Project Structure

```text
Collage_Complaint_Management/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── new_complaint.html
│   ├── complaint_detail.html
│   ├── admin_dashboard.html
│   └── admin_analytics.html
└── static/
    └── css/
        └── style.css
```

The `venv/` folder contains the local Python virtual environment and should not be committed to Git.

## Prerequisites

Install the following before running the application:

- Python 3
- PostgreSQL
- Git
- A code editor such as Visual Studio Code

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Collage_Complaint_Management
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create the PostgreSQL database

Open PostgreSQL using pgAdmin and create a database named:

```text
campuscare_db
```

Use your own PostgreSQL username and password.

### 5. Configure the database connection

Set the database connection string in your PowerShell terminal:

```powershell
$env:DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/campuscare_db"
```

Replace `YOUR_PASSWORD` with your local PostgreSQL password. Do not publish your real password or connection string.

For a persistent or deployed setup, use environment variables or a properly configured `.env` file that is excluded from Git.

### 6. Initialize the database tables

```powershell
flask --app app init-db
```

### 7. Create an administrator account

```powershell
flask --app app create-admin
```

Follow the prompts to enter the administrator's name, email, and password.

### 8. Start the application

```powershell
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

## Complaint Statuses

| Status | Description |
|---|---|
| Submitted | The complaint has been received. |
| In Progress | The complaint is being handled. |
| Resolved | The complaint has been resolved. |
| Rejected | The complaint has been rejected. |

## Security Notes

- Passwords are stored as hashes rather than plain text.
- Student and administrator access is role-based.
- Database credentials and Flask secret keys must be configured privately.
- Never commit passwords, `.env` files, database dumps, or local database files to a public repository.

## Future Improvements

- Email notifications for complaint updates
- Complaint priority levels
- File attachments and supporting images
- Advanced search and filtering
- Improved mobile responsiveness
- Automated testing

## Project Purpose

CampusCare was developed as an academic mini-project to demonstrate web application development, database integration, authentication, role-based access, CRUD operations, and data visualization.

## License

This project is intended for educational purposes. Add a suitable open-source license if you decide to distribute it publicly.
