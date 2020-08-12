from app import locations

data = locations("latitude","longitude")
db.session.add(data)
db.session.commit()