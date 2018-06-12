from flask import Flask, render_template, request, url_for, redirect
app = Flask(__name__,static_url_path='/static')

#@app.route('/')

#def login():
#   return render_template('login.html')
'''
@app.route('/')
@app.route('/login',methods=['POST'])
def login():
    #error = None
    print("Does this print?")
    if request.method == 'POST':
        print(request.form['username'])
        #if valid_login(request.form['username'],
        #              request.form['password']):
        #   return log_the_user_in(request.form['username'])
        #else:
        #    error = 'Invalid username/password'
    else:
        print("hmmm...")
        print(request.method)
        print(request.form)
        print(request.data)
        print(request.args)
        print(request)
        #print(request.form['username'])
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('login.html', error='error')
'''
#Make an app.route() decorator here
"""
@app.route('/', methods=['GET', 'POST']) #allow both GET and POST requests
def form_example():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        language = request.form.get('language')
        framework = request.form['framework']

        return '''<h1>The language value is: {}</h1>
                  <h1>The framework value is: {}</h1>'''.format(language, framework)

    return '''<form method="POST">
                  Language: <input type="text" name="language"><br>
                  Framework: <input type="text" name="framework"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''

#"""


#------------------------------------------------------------------------------

@app.route("/index", methods = ['GET','POST'])
def index():
    return render_template('index.html', error='error')

#------------------------------------------------------------------------------

@app.route("/access_denied", methods = ['GET','POST'])
def access_denied():
    return render_template('access_denied.html', error='error')

#------------------------------------------------------------------------------

def get_function():
    return render_template('login.html', error='error')

#------------------------------------------------------------------------------

def review_objects_get_function():
    return render_template('review_objects_wizard.html',error='error')

def review_objects_post_function():
    print(request.form)
    print(request.method)
    print(request.data)
    print(request.args)
    return redirect(url_for('index'))


@app.route("/object_review",methods=['GET','POST'])
def review_objects():
    if request.method == 'GET':
        return review_objects_get_function()
    elif request.method == 'POST':
        return review_objects_post_function()

#------------------------------------------------------------------------------

@app.route("/under_construction",methods=['GET','POST'])
def under_construction():
    return render_template('under_construction.html',error='error')

#------------------------------------------------------------------------------

@app.route("/load_manual_missions",methods=['GET','POST'])
def load_manual_missions_get_function():
    return render_template('load_mission_wizard.html',error='error')

def load_manual_missions_function():
    print(request.form)
    print(request.method)
    print(request.data)
    print(request.args)
    return redirect(url_for('index'))


@app.route("/object_review",methods=['GET','POST'])
def load_manual_missions():
    if request.method == 'GET':
        return load_manual_missions_get_function()
    elif request.method == 'POST':
        return load_manual_missions_post_function()




def post_function():
    print(request.form)
    print(request.method)
    print(request.form['username'])
    print(request.data)
    print(request.args)
    return redirect(url_for('index'))


@app.route("/login", methods = ['GET', 'POST'])
def login():
    if request.method == 'GET':
        return get_function()

    elif request.method == 'POST':
        return post_function()

#'''
if __name__ == '__main__':
   app.run(debug = True)