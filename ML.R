install.packages("recommenderlab")
install.packages("rrecsys")
install.packages("reshape2")
install.packages("data.table")
install.packages("pvsR")
install.packages("pdp")
install.packages("ggplot2")
install.packages("proxy")

library(recommenderlab)
library(dplyr)
library(ggplot2)
library(reshape2)
library(data.table)
library(pdp)        ### for plotting
library(gridExtra)  ### for plotting
 
setwd("~/Downloads/advanced sl/IMDB-Dataset")
movies=read.csv("movies.csv")
ratings=read.csv("ratings.csv")
head(movies)
head(ratings) 



######### prerpocess of data #####################
#https://data-flair.training/blogs/data-science-r-movie-recommendation/ with modifying
movie_type<- as.data.frame(movies$genres)
movie_type <- as.data.frame(tstrsplit(movie_type[,1], '[|]', 
                                        type.convert=TRUE))
unique(unlist(movie_type))
type <- c("Adventure", "Comedy","Action","Drama","Crime", "Children","Mystery", 
                "Animation",  "Documentary",  "Thriller","Horror","Fantasy","Western",
                "Film-Noir", "Romance","Sci-Fi", "Musical", "War", "IMAX" )
mat1_type <- matrix(0,10330,19)
colnames(mat1_type) <- type
mat1_type[1,] <- type


for (index in 1:nrow(movie_type)) {
  for (col in 1:ncol(movie_type)) {
    gen_col = which(mat1_type[1,] == movie_type[index,col]) 
    mat1_type[index+1,gen_col] <- 1
  }
}
id_withtype<- as.data.frame(mat1_type[-1,]) 

movie_withtype <- cbind(movies[,1:2], id_withtype[])
head(movie_withtype) ### for content based model

#Convert rating matrix into a recommenderlab sparse matrix
#dcast() function in the reshape2 package transforms it.
realRatings <- dcast(ratings, userId~movieId, value.var = "rating", na.rm=FALSE)
realRatings <- as.matrix(realRatings[,-1]) #remove userIds
realRatings <- as(realRatings, "realRatingMatrix")#Objects from the Class




############## exploratory descriptions ############# 
movie_counts <- colCounts(realRatings) # count views for each movie
topmovie<-sort(movie_counts,decreasing = TRUE)
topmovie<-topmovie[1:6] # movieid+ movie_counts
topmovie<-data.frame(movieId=names(topmovie),count=topmovie)
topmovie=merge(topmovie,movies)
topmovie

######## plot #########
#https://github.com/tarashnot/SVDApproximation/blob/master/R/visualize_ratings.R with modifying
visualize_ratings <- function(realRatings,table_views){
  
  data_plot <- data.frame(table(getRatings(realRatings)))
  names(data_plot) <- c("Score", "Count")
  #plot 1
  p1<- ggplot(data_plot,aes(Score,Count)) + geom_bar(stat="identity",fill="skyblue3") +
    geom_text(aes(label=Count),size=3) +
    labs(subtitle = "Count of different ratings") 
  #plot2
  p2<- ggplot(topmovie, aes(x = movieId, y = count)) +
    geom_bar(stat="identity", fill = 'skyblue3') +
    geom_text(aes(label=count), vjust=-0.3, size=2) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(subtitle="Total Views of the Top Films")
   #plot3
  data_plot<- data.frame(rowMeans(realRatings))
  names(data_plot) <- "UserAverageScore"
  p3 <- ggplot(data_plot,aes(x = UserAverageScore)) +
    geom_histogram(binwidth=0.1,color="white", fill = "rosybrown3") +
    geom_vline(xintercept=median(data_plot$UserAverageScore), color = "black", size=0.5) +
    labs(subtitle = " Users' Average Ratings", x = "User Avearge Score", y = "Count") +
    theme(axis.text =element_text(size=5), axis.title=element_text(size=8)) +
    scale_x_continuous(breaks = c(1:5, round(median(data_plot$UserAverageScore), 2)))
  
  #plot4
  data_plot <- data.frame(colMeans(realRatings))
  names(data_plot) <- "ItemAverageScore"
  p4 <- ggplot(data_plot,aes(x = ItemAverageScore)) +
    geom_histogram(binwidth=0.1, colour = "white", fill = "rosybrown3") +
    geom_vline(xintercept=median(data_plot$ItemAverageScore), color = "black", size=0.5) +
    labs(subtitle = " Items' Average Ratings", x = "Item Avearge Score", y = "Count") +
    theme(axis.text =element_text(size=5), axis.title=element_text(size=8)) +
    scale_x_continuous(breaks = c(1:5, round(median(data_plot$ItemAverageScore), 2)))
  
  return(grid.arrange(arrangeGrob(p1,p2, ncol=2), arrangeGrob(p3, p4, ncol=2), nrow = 2))
}
visualize_ratings(realRatings,topmovie)
a<-rowCounts(realRatings)
summary(a)

#################### model  ######################
#create 80/20 split for the data
set.seed(56872)
e <- evaluationScheme(realRatings, method="split", train=0.8,given=-5,goodRating=4)
#for certain subject in test 5 will be unknown values
train <- getData(e, 'train')
test_known <- getData(e, 'known')
test_unknown <- getData(e, 'unknown')


###################### baseline random ##########################
set.seed(56872)
m1 <- Recommender(train,method='RANDOM')
# Make item recommendations for users
m1class <- predict(m1, test_known, n=5, type='topNList')
# m1class <- as(m1class, 'list') for checking specific recommended item
# Predict ratings
m1rating <- predict(m1, test_known,type='ratings')
# m1rating <- as(m1rating, 'list')
# Compute metrics for topN results
perf_m1class <- calcPredictionAccuracy(m1class, test_unknown, given=-5,goodRating=4)
# compute metrics based on rating prediction
# metrics averaged over all recommendations
perf_m1rating<- calcPredictionAccuracy(m1rating ,test_unknown,byUser=F)
perf_m1class
perf_m1rating

######################## popular ##########################
set.seed(56872)
m2<- Recommender(train,method='POPULAR',param=list(normalize = "center"))
# Make item recommendations for users
m2class <- predict(m2, test_known,n=5, type='topNList')
# Explore the returned prediction for the first
#m2class <- as(m2class, 'list')

# Predict ratings
m2rating <- predict(m2, test_known,type='ratings')
# Show the predicted rating as a list
#m2rating <- as(m2rating, 'list')
# Compute metrics for topN results
perf_m2class <- calcPredictionAccuracy(m2class,test_unknown,given=-5,goodRating=4)
# compute metrics based on rating prediction
perf_m2rating <- calcPredictionAccuracy(m2rating,test_unknown,byUser=F)
perf_m2class
perf_m2rating

############################# user-basedcollaborative filtering ##########################
# http://studio.galaxystatistics.com/report/recommenderlab/article1/  
# reference to evaluation function but with modification
#UBCF: classification-nn=80 rating-nn=100
set.seed(56872)
hyperdata <- evaluationScheme(realRatings, method="split", train=0.7,given=-5,goodRating=4)
algorithms <- list(
  "UBCF80" = list(name="UBCF", param=list(nn=80)), 
  "UBCF90" = list(name="UBCF", param=list(nn=90)),
  "UBCF100" = list(name="UBCF", param=list(nn=100)),
  "UBCF105" = list(name="UBCF", param=list(nn=105)),
  "UBCF110" = list(name="UBCF", param=list(nn=110))
 )

classresults <- evaluate(hyperdata, method = algorithms,type = "topNList",
                         n=c(1,seq(10, 100, 10)))
plot(classresults, legend="topleft")#ROC ->80

ratingresults <- evaluate(hyperdata, method = algorithms, type ="ratings")
plot(ratingresults) #NN=100


########### predict on original dataset
set.seed(56872)
m31 <- Recommender(train, method = "UBCF",
                   param=list(normalize = "center", method="Cosine", nn=80))
m3class  <- predict(m31, test_known, n=5,type="topNList")
perf_m3class <- calcPredictionAccuracy(m3class,test_unknown,given=-5,goodRating=4)

m32 <- Recommender(train, method = "UBCF",
                   param=list(normalize = "center", method="Cosine", nn=105))
m3ratings  <- predict(m32, test_known, type="ratings")
perf_m3rating <- calcPredictionAccuracy(m3ratings,test_unknown)

perf_m3class
perf_m3rating

############################# item-basedcollaborative filtering ##########################
# tune hyperparameters
set.seed(56872)
hyperdata <- evaluationScheme(realRatings, method="split", train=0.7,given=-5,goodRating=4)
algorithms <- list(
  "IBCF10" = list(name="IBCF", param=list(nn=10)), 
  "IBCF20" = list(name="IBCF", param=list(nn=20)),
  "IBCF30" = list(name="IBCF", param=list(nn=30)),
  "IBCF40" = list(name="IBCF", param=list(nn=40)),
  "IBCF50" = list(name="IBCF", param=list(nn=50))
)

classresults <- evaluate(hyperdata, method = algorithms,type = "topNList",
                         n=c(1,seq(10, 100, 10)))
plot(classresults, legend="bottomright")
ratingresults <- evaluate(hyperdata, method = algorithms, type ="ratings")
plot(ratingresults)

########### predict on original dataset
set.seed(56872)
m41 <- Recommender(train, method = "IBCF",
                   param=list( normalize = "center",method="Cosine", k=30))
m4class  <- predict(m41, test_known, n=5,type="topNList")
m4ratings  <- predict(m41, test_known, type="ratings")

perf_m4class <- calcPredictionAccuracy(m4class,test_unknown,given=-5,goodRating=4)
perf_m4rating <- calcPredictionAccuracy(m4ratings,test_unknown)


############################# content-based model ##########################
#https://muffynomster.wordpress.com/2015/06/07/building-a-movie-recommendation-engine-with-r/ 
#refer with modifications self-made function

ratings$binrating<-ifelse(ratings$rating>3,1,-1)
biratings <- dcast(ratings, movieId~userId, value.var = "binrating", na.rm=FALSE)
biratings[is.na(biratings)] <- 0
biratings = biratings[,-1] # Rows are movieIds, cols are userIds
id_withtype <- data.frame(lapply(id_withtype,function(x){as.integer(x)-1}))
id_withtype <- id_withtype[-which((movies$movieId %in% ratings$movieId) == FALSE),]
#dot product to build up user profile
max_idwithtype<-data.matrix(id_withtype)
max_biratings<-data.matrix(biratings)
tmax_idwithtype<-t(max_idwithtype)
user_profile<-tmax_idwithtype %*% max_biratings
user_profile<-data.frame(user_profile)
user_profile<- data.frame(lapply(user_profile,function(x){ifelse(x<0,0,1)}))
#column -- userId  , row--genre
tmax_idwithtype1<-tmax_idwithtype
rownames(tmax_idwithtype1)<-NULL
#Calculate Jaccard distance between user profile and all movies
jaccard <- function(a, b) {
  ncount=0
  for (i in 1:19) {
    if (a[i]==b[i]){
      ncount=ncount+1
    }
  }
  return (ncount/19)
}

#668 users 10325 movies
recommend_matrix<-matrix(0,668,5)
for (i in 1:668) {
  user <- user_profile[,i] #user_profile 19*668
  jaccard_score<-NULL
  for(j in 1:ncol(tmax_idwithtype1)){#tmax_idwithtype1 19*10325
    jaccard_score[j]=jaccard(user,tmax_idwithtype1[,j])
  }  
  names(jaccard_score)<-c(1:10325)
  top5<-sort(jaccard_score,decreasing = TRUE)
  recommend_matrix[i,]<-as.integer(names(top5[1:5]))
}
head(recommend_matrix)
#calcaulte precision
content_precision<-function(a,b){
  countprecision=0
  for (i in 1:668) {
    countprecision=countprecision+length(intersect(a[,i],b[i,]))/5
  }
  return(countprecision/668)
}
content_precision(biratings,recommend_matrix) #0.0005988024

############################# SVD ##############################
# tune hyperparameters
set.seed(56872)
hyperdata <- evaluationScheme(realRatings, method="split", train=0.7,given=-5,goodRating=4)
algorithms <- list(
  "SVD10" = list(name="SVD", param=list(k=10,maxiter=500,normalize='center')), 
  "SVD15" = list(name="SVD", param=list(k=15,maxiter=500,normalize='center')),
  "SVD20" = list(name="SVD", param=list(k=20,maxiter=500,normalize='center')),
  "SVD25" = list(name="SVD", param=list(k=25,maxiter=500,normalize='center')),
  "SVD30" = list(name="SVD", param=list(k=30,maxiter=500,normalize='center'))
)

classresults <- evaluate(hyperdata, method = algorithms,type = "topNList",
                         n=c(1,seq(10, 100, 10)))
plot(classresults, legend="bottomright")
ratingresults <- evaluate(hyperdata, method = algorithms, type ="ratings")
plot(ratingresults)

########### predict on original dataset
set.seed(56872)
m5<- Recommender(train, method='SVD',
                        param=list(k=15,maxiter=500,normalize='center'))
m5class<- predict(m5, test_known,n=5, type='topNList')
m5ratings<- predict(m5, test_known,type='ratings')

perf_m5class <- calcPredictionAccuracy(m5class,test_unknown,given=-5,goodRating=4)
perf_m5rating <- calcPredictionAccuracy(m5ratings,test_unknown)


############################# FunkSVD ##########################
# tune hyperparameters
set.seed(56872)
algorithms <- list(
  "SVDF10" = list(name="SVDF", param=list(k=10,lambda=0.001,
                                          max_epochs=50,normalize='center')), 
  "SVDF15" = list(name="SVDF", param=list(k=15,lambda=0.001,
                                          max_epochs=50,normalize='center')),
  "SVDF20" = list(name="SVDF", param=list(k=20,lambda=0.001,
                                          max_epochs=50,normalize='center'))
)

classresults <- evaluate(hyperdata, method = algorithms,type = "topNList",
                         n=c(1,seq(10, 100, 10)))
plot(classresults, legend="bottomright")
ratingresults <- evaluate(hyperdata, method = algorithms, type ="ratings")
plot(ratingresults)

########### predict on original dataset
set.seed(56872)
m6<- Recommender(train,method='SVDF',
                         param=list(k=15,lambda=0.001,
                                    max_epochs=50,normalize='center'))
m6class<- predict(m6, test_known,n=5, type='topNList')
m6ratings<- predict(m6, test_known,type='ratings')

perf_m6class <- calcPredictionAccuracy(m6class,test_unknown,given=-5,goodRating=4)
perf_m6rating <- calcPredictionAccuracy(m6ratings,test_unknown)

############################# Hybrid system ##########################
set.seed(56872)
m7<- HybridRecommender(m2,m5,m6,
                       weights=c(0.2, 0.3, 0.5))
m7class <- predict(m7, test_known,n=5, type='topNList')
m7ratings<- predict(m7, test_known,type='ratings')

perf_m7class <- calcPredictionAccuracy(m7class,test_unknown,given=-5,goodRating=4)
perf_m7rating <- calcPredictionAccuracy(m7ratings,test_unknown)
perf_m7class 
perf_m7rating




