import pandas as pd,datetime,numpy as np,plotly.graph_objs as go,re
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
url = 'https://github.com/neylsoncrepalde/projeto_eda_covid/blob/master/covid_19_data.csv?raw=true'
df = pd.read_csv(url,parse_dates=['ObservationDate', 'Last Update'])

def crr(Cnomes):return re.sub(r'[/| ]','',Cnomes).lower();
df.columns = [crr(col) for col in df.columns]

br = df.loc[(df.countryregion == 'Brazil')&(df.confirmed > 0)]
br['novoscasos'] = list(map(
    lambda x: 0 if(x==0) else br['confirmed'].iloc[x]-br['confirmed'].iloc[x-1],
    np.arange(br.shape[0])
   ))

def TC(data,var,Dinicio=None,Dfim=None):
    if Dinicio ==None:
        Dinicio = data.observationdate.loc[data[var]> 0 ].min()
    else:Dinicio = pd.to_datetime(Dinicio)
    if Dfim ==None:
        Dfim = data.observationdate.iloc[-1]
    else:Dfim = pd.to_datetime(Dfim)
    passado = data.loc[data.observationdate == Dinicio,var].values[0]
    presente = data.loc[data.observationdate == Dfim, var].values[0]
    n = ((Dfim - Dinicio).days)
    tx = (presente/passado)**(1/n)-1
    return tx*100
def TCD(data,var,Dinicio=None):
    if Dinicio ==None:
        Dinicio = data.observationdate.loc[data[var]> 0 ].min()
    else:Dinicio = pd.to_datetime(Dinicio)
    Dfim = data.observationdate.max()
    n = ((Dfim - Dinicio).days)
    tx = list(map(
        lambda x:(data[var].iloc[x]-data[var].iloc[x-1])/data[var].iloc[x-1],
        range(1,n+1)
        ))
    return np.array(tx)*100
Media = TC(br,'confirmed');Diaria = TCD(br,'confirmed')
D1 = br.observationdate.loc[br.confirmed > 0].min()
confirmado = br.confirmed;confirmado.index = br.observationdate
mod = auto_arima(confirmado);mod2 = auto_arima(Diaria)
fig = go.Figure(go.Scatter(x=confirmado.index,y=confirmado,name='Casos já Observados'))
fig.add_trace(go.Scatter(x=pd.date_range('2020-05-20','2021-10-29'),y=mod.predict(527),name='previsão dos casos'))
fig.add_trace(go.Scatter(x=pd.date_range(D1,br.observationdate.max())[1:],y=Diaria,name='% contaminação'))
fig.add_trace(go.Scatter(x=pd.date_range('2020-05-20','2021-10-29'),y=mod2.predict(527),name='previsão % contaminação'))
fig.update_layout(title="previsão de casos e contaminação.Media %"+str(Media))
fig.show()
