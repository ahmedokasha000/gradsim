from flask import Flask
from flask_sqlalchemy import SQLAlchemy 
from flask import render_template

app= Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']= False
app.config["SQLALCHEMY_DATABASE_URI"]='postgresql://admin:2020@localhost:5432/bump'
db = SQLAlchemy(app)


class locations(db.Model):

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    latitude = db.Column(db.String(50), unique=False, nullable=False)
    longitude = db.Column(db.String(50), unique=False, nullable=False)

    def __init__(self , latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

@app.route('/list_all_user')
def ListAllUsers():
    location= locations.query.all()
    return render_template('list_all_user.html', mylocations=location)



if __name__ =='__main__':
    
    app.run()
