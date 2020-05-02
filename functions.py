class Timer():
    
    """Timer class designed to keep track of and save modeling runtimes. It
    will automatically find your local timezone. Methods are .stop, .start,
    .record, and .now"""
    
    def __init__(self, fmt="%m/%d/%Y - %I:%M %p", verbose=None):
        import tzlocal
        self.verbose = verbose
        self.tz = tzlocal.get_localzone()
        self.fmt = fmt
        
    def now(self):
        import datetime as dt
        return dt.datetime.now(self.tz)
    
    def start(self):
        if self.verbose:
            print(f'---- Timer started at: {self.now().strftime(self.fmt)} ----')
        self.started = self.now()
        
    def stop(self):
        print(f'---- Timer stopped at: {self.now().strftime(self.fmt)} ----')
        self.stopped = self.now()
        self.time_elasped = (self.stopped - self.started)
        print(f'---- Time elasped: {self.time_elasped} ----')
        
    def record(self):
        try:
            self.lap = self.time_elasped
            return self.lap
        except:
            return print('---- Timer has not been stopped yet... ----')
        
    def __repr__(self):
        return f'---- Timer object: TZ = {self.tz} ----'

def fit_n_pred(clf_, X_tr, X_te, y_tr):
    
    """Takes in Classifier, training data (X,y), and test data(X). Will output 
    predictions based upon both the training and test data using the sklearn
    .predict method. MUST unpack into two variables (train, test)."""
    
    clf_.fit(X_tr, y_tr)

    y_hat_trn = clf_.predict(X_tr)
    y_hat_tes = clf_.predict(X_te)
    
    display(clf_)
    return y_hat_trn, y_hat_tes

def plot_importance(tree, X_tr, top_n=10, figsize=(10,10), ax=None):
    
    """Takes in pre-fit descision tree and the training X data used. Will output
    a horizontal bar plot (.plt) of the top 10 (default) features used in said tree."""
    
    import pandas as pd
    import matplotlib as plt

    imps = pd.Series(tree.feature_importances_,index=X_tr.columns)
    imps.sort_values(ascending=True).tail(top_n).plot(kind='barh',figsize=figsize, ax=ax)
    return imps

def evaluate_model(clf_, X_tr, X_te, y_tr, y_te, cls_rpt_tr=False, show=True, cls_labels=None, binary=False):
    
    
    """Takes any classifier, train/test data for X/y, labels for graph (optional).
    Will output (if show) a Sklearn Classification Report and Confusion Matrix
    along with a Yellowbrick ROC/AUC curve and Feature Importance graph (if a tree).
    Otherwise will return training/test predictions."""
    
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    from yellowbrick.classifier import ROCAUC
    
    ## Fit and predict 
    y_hat_trn, y_hat_tes = fit_n_pred(clf_, X_tr, X_te, y_tr)
    
    if show:
        ## Classification Report / Scores
        if cls_rpt_tr:
            print('Classification Report Train')
            print(metrics.classification_report(y_tr,y_hat_trn))
        else:
            print('Classification Report Test')
            print(metrics.classification_report(y_te,y_hat_tes))

        ## Confusion Matrix
        fig, ax = plt.subplots(figsize=(10,5), ncols=2)
        
        metrics.plot_confusion_matrix(clf_,X_te,y_te,cmap="Greens",
                                      normalize='true',ax=ax[0])
        ax[0].set(title='Confusion Matrix Test Data')
        ax[0].grid(False)        

        roc = ROCAUC(clf_, classes=cls_labels, ax=ax[1])
        roc.fit(X_tr, y_tr)
        roc.score(X_te, y_te)
        roc.finalize()
            
        plt.tight_layout()
        plt.show()
        
        if binary:
            try:
                imps = plot_importance(clf_, X_tr)
            except:
                imps = None
        
    else:
        return y_hat_trn, y_hat_tes

def time_models(x_tr, x_te, y_tr, y_te, mod_list, mod_labels, time_obj, count=0, keep=True, show=True, cls_lab=None):
    
    """Takes in X/y data (train and test), list of model(s) and Timer() object.
    Optional parameters determine (with keep=True) which label to start with 
    (count) and (with show=True) what class labels to display (cls_lab). Outputs
    visuals from evaluate_model() and duration dictionary if opted-in."""
    
    durations = {}
    
    for m in mod_list:
        if show:
            time_obj.start()
            evaluate_model(m, x_tr, x_te, y_tr, y_te, cls_labels=cls_lab)
            time_obj.stop()
        else:
            time_obj.start()
            a, b = evaluate_model(m, x_tr, x_te, y_tr, y_te, show=False)
            time_obj.stop()
        
        if keep:
            durations[mod_labels[count]] = time_obj.record().total_seconds()
        
        count += 1
    
    if keep:
        return durations
    
    else:
        return f'---- All done! Completed {count} models. ----'

def summarize_model(clf_, X_tr, X_te, y_tr, y_te, tree=False):
    
    """Takes any classifier and train/test data for X/y. Will output Sklearn
    Classification Report and Confusion Matrix. If classifier is decision tree
    like, can select for a feature importance graph."""
    
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    
    y_hat_tr, y_hat_te = fit_n_pred(clf_, X_tr, X_te, y_tr)
    print('Classification Report:')
    print(metrics.classification_report(y_te, y_hat_te))
    
    if tree:
        fig, ax = plt.subplots(figsize=(10,5), nrows=2)

        metrics.plot_confusion_matrix(clf_,X_te,y_te,cmap="Greens", normalize='true',
                                     ax=ax[0])
        ax[0].set(title='Confusion Matrix')
        ax[0].grid(False)

        plot_importance(clf_, X_tr, ax=ax[1])
        plt.tight_layout()
        
    else:
        metrics.plot_confusion_matrix(clf_,X_te,y_te,cmap="Greens", normalize='true')
        plt.title('Confusion Matrix')
        plt.grid(False)
        plt.tight_layout()

def grid_searcher(clf_, params, X_tr, X_te, y_tr, y_te, cv=None, keep_t=False, train_score=True):
    
    """Takes any classifier, train/test data for X/y, and dict of parameters to
    iterate over. Optional parameters select for cross-validation tuning, keeping
    time for running the gridsearch, and returning training scores when done.
    Default parameters only return the fitted grid search object. MUST HAVE Timer
    class imported."""
    
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    
    ## Instantiate obj. with our targets
    grid_s = GridSearchCV(clf_, params, cv=cv, return_train_score=train_score)
    
    ## Time and fit run the 'search'
    time = Timer()
    time.start()
    grid_s.fit(X_tr, y_tr)
    time.stop()
    
    ## Display results
    tr_score = np.mean(grid_s.cv_results_['mean_train_score'])
    te_score = grid_s.score(X_te, y_te)
    print(f'Mean Training Score: {tr_score :.2%}')
    print(f'Mean Test Score: {te_score :.2%}')
    print('Best Parameters:')
    print(grid_s.best_params_)
    
    ## Time keeping and grid obj
    if keep_t:
        lap = time.record().total_seconds()
        print('**********All done!**********')
        return grid_s, lap
    else:
        return grid_s

def NA_handler_bin(df):
    
    """VERY specific helper function designed to minimize code length in notebook.
    Details the preprocessing steps used to Fill NAs according to specific
    strategies."""

    import pandas as pd
    import numpy as np
    import missingno as mg
    
    ## Check dist
    print('Stance Distributions:')
    print(df['B_Stance'].value_counts(normalize=True), '\n')
    print(df['R_Stance'].value_counts(normalize=True))
    
    ## Making list for random choices
    stance_list = list(df['B_Stance'].dropna().unique())
    print('\nStances:')
    print(stance_list, '\n')

    ## Making array of probabilities for corresp. stances
    print('Stance Probabilities:')
    stance_ps = df['B_Stance'].value_counts(normalize=True).values
    print(stance_ps, '\n')
    
    ## Using for loop to randomly fill stances according to probabilities 
    helper = ['B_Stance', 'R_Stance']
    for col in helper:
        df[col] = df[col].fillna(np.random.choice(stance_list, p=stance_ps))
        
    ## Using .groupby to find aggregrate stats for groups
    height_grouped = df.groupby('R_Height_cms').agg('mean')
    
    ## Making dictionary of reach values acc. to height grouping
    reach_by_height = dict(height_grouped['R_Reach_cms'])
    
    ## Same as above but targeting Height values to be filled
    reach_grouped = df.groupby('R_Reach_cms').agg('mean')
    height_by_reach = dict(reach_grouped['R_Height_cms'])
    
    ## Targeting weight values to be filled
    weight_cls_grouped = df.groupby('weight_class').agg('mean')
    weight_by_class = dict(weight_cls_grouped['R_Weight_lbs'])
    
    ## Mapping groupby aggregates to NAs in corresp. columns
    cols1 = ['B_Reach_cms', 'R_Reach_cms']
    for col in cols1:
        df[col] = df[col].fillna(df['R_Height_cms'].map(reach_by_height))
    cols2 = ['B_Height_cms', 'R_Height_cms']
    for col in cols2:
        df[col] = df[col].fillna(df['R_Reach_cms'].map(height_by_reach))
    cols3 = ['B_Weight_lbs', 'R_Weight_lbs']
    for col in cols3:
        df[col] = df[col].fillna(df['weight_class'].map(weight_by_class))
    
    ## Checking for rows with NAs after mapping
    print('Null Values Remaining:')
    display(df.loc[(pd.isnull(df.R_Reach_cms)), 'R_Reach_cms'])
    display(df.loc[(pd.isnull(df.B_Reach_cms)), 'B_Reach_cms'])
    
    ## Using mapped values according to weight class
    ## Indices taken from above
    val_R_rch2_bin = {4800: 196.393271, 4814: 196.393271}
    val_B_rch2_bin = {4800: 196.227178, 4814: 196.227178}
    cols = ['R_Reach_cms', 'B_Reach_cms']
    
    ## Assigning values acc. to column
    df[cols[0]] = df[cols[0]].fillna(val_R_rch2_bin)

    df[cols[1]] = df[cols[1]].fillna(val_B_rch2_bin)
    
    ## Creating variable to hold age mean for filling by taking midpoint 
    ## between the columns
    print('Age Mean:')
    print(df[['B_age', 'R_age']].mean(), '\n')
    print('Midpoint Age Mean:')
    age_mean = (df['B_age'].mean() + df['R_age'].mean()) / 2
    print(age_mean)

    ## Fill NA values by the mean age overall
    df['B_age'] = df['B_age'].fillna(age_mean)
    df['R_age'] = df['R_age'].fillna(age_mean)
    
    ## Missingno visual check
    display(mg.matrix(df));
    
    return df