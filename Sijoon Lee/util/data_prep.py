import pandas as pd
from pandas import read_csv
from timezonefinder import TimezoneFinder # !pip install timezonefinder
from datetime import datetime
import pytz
import urllib.request
import calendar


RAW_DATA = "data/raw-data"
PROCESSED_DATA = "data/processed-data"
STATION = "data/station"


class Downloader(object):

	def __init__(self, station="47267", station_data_file="Station-Inventory-EN.csv"):
		self.station = station
		self.station_data_file = station_data_file
		self.latitude, self.longitude = self.getCoordinate(self.station, self.station_data_file)
		self.timezone = self.getTimezone()

	def getCoordinate(self, station="47267", filename = "Station-Inventory-EN.csv"):
		df = read_csv("{STATION}/{filename}".format(STATION=STATION, filename=filename, header= 0 ))

		stationInfo = df.loc[df[df.columns[3]] == int(station)]

		latitude = stationInfo.iat[0, 6]
		longitude = stationInfo.iat[0, 7]
		return latitude, longitude

	def getTimezone(self):
		tf = TimezoneFinder(in_memory=True)
		return tf.timezone_at(lng=self.longitude, lat=self.latitude) # America/Toronto

	def getTimeAtStation(self):
		station_now = datetime.now(pytz.timezone(self.timezone))
		return [station_now.year, station_now.month, station_now.day, station_now.hour, station_now.minute]

	def download(self, year, month, station, filename):
		'''
			Download weather data on specific month and year at given station ID 
			Arg : year, month, stationID(47267 = Kingston Climate Station)
			Return : none
			Output : {stationID}-{year}-{month}.csv file under /raw-data directory
			c.f.: https://stackoverflow.com/questions/50260574/wget-content-disposition-using-python
		'''
		url = "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={station}&Year={year}&Month={month}&Day=14&timeframe=1&submit=Download+Data".format(year=year, month=month, station=station)
		urllib.request.urlretrieve(url, filename)

	def generateFilename(self, year, month, station):
		filename = "{RAW_DATA}/{station}-{year}-{month:02d}.csv".format(\
			RAW_DATA = RAW_DATA, station=station, year=year, month=month)
		return filename

	def downloadFromTo(self, fromWhen=[2015,1], toWhen=[2019,7], station="47267"):
		filenames = []

		if fromWhen[0] < toWhen[0]:

			# download the first year
			for month in range(fromWhen[1], 13):
				filename = self.generateFilename(fromWhen[0], month, station)
				filenames.append(filename)
				self.download(fromWhen[0], month, station, filename)

			# download the middle years
			for year in range(fromWhen[0]+1, toWhen[0]):
				for month in range(1,13):
					filename = self.generateFilename(year, month, station)
					filenames.append(filename)
					self.download(year, month, station, filename)

			# download the last year
			for month in range(1, toWhen[1]+1):
				filename = self.generateFilename(toWhen[0], month, station)
				filenames.append(filename)
				self.download(toWhen[0], month, station, filename)

		elif fromWhen[0] is toWhen[0]:
			for month in range(fromWhen[1], toWhen[1]+1):
				filename = self.generateFilename(fromWhen[0], month, station)
				filenames.append(filename)
				self.download(fromWhen[0], month, station, filename)

		else:
			print("use vaild years")

		return filenames



	def createDataFrame(self, filenames):
		# load csv files and combine
		df = read_csv(filenames[0], skiprows = 15, header = 0)
		for count in range(1, len(filenames)):
			df = pd.concat([df, read_csv(filenames[count], skiprows = 15, header = 0)])
		return df

	# https://stackoverflow.com/questions/15891038/change-data-type-of-columns-in-pandas/44536326
	# ex) 01:00 -> 1.0
	def convertHourToFloat(self, df):
		df[df.columns[4]] = df[df.columns[4]].str.slice(0,2).astype("float64")
		return df

	def deleteColumns(self, df, to_be_deleted = [1, 3, 6, 8, 10, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23] ):
		df = df.drop(df.columns[to_be_deleted], axis=1)
		return df.set_index(df.columns[0])
		
	def interpolateNA(self, df, limit=500):
		
		df = df.interpolate(method='linear', limit_direction ='both', limit=500)
		#print(df.isnull().sum())
		return df

	def temperatureFirst(self, df):
		cols = df.columns.tolist()
		cols = cols[2:] + cols[:2]
		df = df[cols]
		return df
	'''
	if df contains dummy data on and after current day, this function truncates df
	if today is 2019-8-8, this functioon delete dummy data from 2019-8-8 to the last day of the month

	argument
		df: pandas dataframe
		toWhen: array that contains [year, month]
		now_at_station: array that contains current [year, month, day, hour, minute]
	return
		
		truncated dataframe
	'''
	def truncateBack(self, df, toWhen, now_at_station):
		if toWhen[0] == now_at_station[0] and toWhen[1] == now_at_station[1]:
			lastDay = calendar.monthrange(now_at_station[0], now_at_station[1])[1]
			df = df[:len(df) - (lastDay - now_at_station[2] + 1) * 24 ]
		return df

	def saveDataFrame(self, df, filename="export.csv"):
		df.to_csv (r'{PROCESSED_DATA}/{filename}'.format(PROCESSED_DATA=PROCESSED_DATA, filename=filename, header=True))

	def buildDataFrame(self, fromWhen = [2015,1], toWhen = [2019,8], station = "47267", save = True, saveFile = "example.csv"):
		files = self.downloadFromTo(fromWhen,toWhen, station)
		df = self.createDataFrame(files)
		df = self.convertHourToFloat(df)
		df = self.deleteColumns(df)
		now_at_station = self.getTimeAtStation()
		df = self.truncateBack(df, toWhen, now_at_station)
		df = self.interpolateNA(df)
		df = self.temperatureFirst(df)
		if save:
			self.saveDataFrame(df, saveFile)
		return df

# print(buildDataFrame([2015,1], [2019,8], "47267", True, "modelSequence.csv")[:3])

	def getLatestData(self, station = "47267", length = 480, save = True, saveFile = "latestSequence.csv"):
		assert length < 1000,\
			print("don't use too long sequence, lengh should be lesser than 1000")
		
		year, month, day, _, _ = self.getTimeAtStation()
		if (day-1)*24 < length:
			if month is not 1:
				df = self.buildDataFrame([year, month-1], [year, month], station, False)
			if month is 1:
				df = self.buildDataFrame([year - 1, 12], [year, month], station, False)
		
		df = df[-length-23:]
		if save:
			self.saveDataFrame(df, saveFile)
		
		return df, year, month, day
