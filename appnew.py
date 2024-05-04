from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Read the CSV file
data = pd.read_csv("student_data.csv")

X = data[['student_cgpa', 'coding_score', 'academic_performance_score', 'personality_test_score']]
y = data[['student_id']]
model = RandomForestRegressor()
model.fit(X, y)


avg_cgpa = data['student_cgpa'].mean()
avg_coding_score = data['coding_score'].mean()
avg_academic_performance = data['academic_performance_score'].mean()
avg_personality_test = data['personality_test_score'].mean()

# Function to calculate Section to Improve
def section_to_improve(row):
    sections = []
    if row['student_cgpa'] < avg_cgpa:
        sections.append('CGPA')
    if row['coding_score'] < avg_coding_score:
        sections.append('Coding')
    if row['academic_performance_score'] < avg_academic_performance:
        sections.append('Academic Performance')
    if row['personality_test_score'] < avg_personality_test:
        sections.append('Personality Test')
    return ', '.join(sections) if sections else "Good going keep practicing the same way"

# Generate suggested topics based on sections to improve
def suggest_topics(row):
    topics = []
    if 'CGPA' in row['Section_To_Improve']:
        topics.append("Work on the college core subjects")
    if 'Coding' in row['Section_To_Improve']:
        topics.append("Practice DSA, Logic building")
    if 'Academic Performance' in row['Section_To_Improve']:
        topics.append("Analytical and logical topics")
    if 'Personality Test' in row['Section_To_Improve']:
        topics.append("Work on time management and on yourself to utilize full of yourself")
    return ', '.join(topics)

# Apply functions to each row
data['Section_To_Improve'] = data.apply(section_to_improve, axis=1)
data['Suggested_Topics'] = data.apply(suggest_topics, axis=1)



# Route to render index.html
@app.route("/")
def index():
    return render_template("index.html")

def generate_student_html(student_data):
    if student_data.empty:
        return "<p>Student data not found.</p>"
    else:
        student_html = "<h2>Student Details</h2>"
        student_html += "<div class='section'><h3 class='section-heading'>Student name:</h3><div class='section-details'>{}</div></div>".format(student_data.iloc[0]['student_name'])
        student_html += "<div class='section'><h3 class='section-heading'>Student id:</h3><div class='section-details'>{}</div></div>".format(student_data.iloc[0]['student_id'])

        student_html += "<h2>Performance</h2>"
        student_html += "<div class='section'><h3 class='section-heading'>CGPA:</h3><div class='section-details'>{}</div></div>".format(student_data.iloc[0]['student_cgpa'])
        student_html += "<div class='section'><h3 class='section-heading'>Coding Score:</h3><div class='section-details'>{}</div></div>".format(student_data.iloc[0]['coding_score'])
        student_html += "<div class='section'><h3 class='section-heading'>Academic Performance:</h3><div class='section-details'>{}</div></div>".format(student_data.iloc[0]['academic_performance_score'])
        student_html += "<div class='section'><h3 class='section-heading'>Personality Score:</h3><div class='section-details'>{}</div></div>".format(student_data.iloc[0]['personality_test_score'])

        student_html += "<h2>Analysis</h2>"
        student_html += "<div class='section'><h3 class='section-heading'>Section to Improve:</h3><div class='section-details'>{}</div></div>".format(student_data.iloc[0]['Section_To_Improve'])
        student_html += "<div class='section'><h3 class='section-heading'>Suggested Topics:</h3><div class='section-details'>{}</div></div>".format(student_data.iloc[0]['Suggested_Topics'])
        student_html += "<div class='section'><h3 class='section-heading'>Overall Performance:</h3><div class='section-details'>{}</div></div>".format(student_data.iloc[0]['Performance_Status'])

        return student_html


# Route to validate student login
def validate_student():
    student_name = request.form["student_name"]
    student_id = int(request.form["student_id"])
    student_data = data[(data['student_name'] == student_name) & (data['student_id'] == student_id)]
    if not student_data.empty:
        X_pred = student_data[['student_cgpa', 'coding_score', 'academic_performance_score', 'personality_test_score']]
        performance_status = round(model.predict(X_pred)[0])   # Round to two decimal places
        # performance_status = int(performance_status)  # Convert to integer
        student_data['Performance_Status'] = performance_status
        return generate_student_html(student_data)
    else:
        return "<p>Student data not found. Please check your credentials.</p>"

# Route to get details for a specific student by student ID
@app.route("/student/<int:student_id>")
def get_student(student_id):
    student_data = data[data['student_id'] == student_id]
    if not student_data.empty:
        X_pred = student_data[['student_cgpa', 'coding_score', 'academic_performance_score', 'personality_test_score']]
        performance_status = model.predict(X_pred)[0]
        student_data['Performance_Status'] = performance_status
        return generate_student_html(student_data)
    else:
        return "<p>Student data not found.</p>"

# Route to get details for all students
@app.route("/all")
def get_all_students():
    all_students_html = "<table class='table'><thead><tr><th>Student Id</th><th>Student Name</th><th>CGPA</th><th>Coding Score</th><th>Academic Performance</th><th>Personality Score</th><th>Section to Improve</th><th>Suggested Topics</th><th>Overall Performance</th></tr></thead><tbody>"
    for index, row in data.iterrows():
        X_pred = [row['student_cgpa'], row['coding_score'], row['academic_performance_score'], row['personality_test_score']]
        performance_status = model.predict([X_pred])[0]
        all_students_html += "<tr>"
        all_students_html += f"<td>{row['student_id']}</td>"
        all_students_html += f"<td>{row['student_name']}</td>"
        all_students_html += f"<td>{row['student_cgpa']}</td>"
        all_students_html += f"<td>{row['coding_score']}</td>"
        all_students_html += f"<td>{row['academic_performance_score']}</td>"
        all_students_html += f"<td>{row['personality_test_score']}</td>"
        all_students_html += f"<td>{row.get('Section_To_Improve', 'N/A')}</td>"
        all_students_html += f"<td>{row.get('Suggested_Topics', 'N/A')}</td>"
        all_students_html += f"<td>{performance_status}</td>"
        all_students_html += "</tr>"
    all_students_html += "</tbody></table>"
    return all_students_html

if __name__ == "__main__":
    app.run(debug=True)
