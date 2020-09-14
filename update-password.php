<?php
    session_start();
?>
<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Update Password</title>
    

    <link rel="stylesheet" href="css/bootstrap.min.css"/>
    
    <link rel="stylesheet" href="fontawesome/css/all.min.css">
    <link rel="stylesheet" href="css/login.css"/>
    
    <script src="js/jquery-3.4.1.min.js"></script>
    <script src="js/bootstrap.min.js"></script>
    
    
</head>
<body>
   <span id="error">

                     <?php
                     if(isset($_SESSION['errors'])){
                         if(!empty($_SESSION['errors'])){
                             echo '<div class="alert alert-danger">';
                             echo '<ul>';
                             foreach ($_SESSION['errors'] as $error){
                                 echo '<li>' ;
                                    echo $error;
                                 echo '</li>' ;
                             }
                             echo '</ul>';
                             echo   '</div>';
        }
        }
        ?>
        </span>
    <form action="update-data.php" method="post">
        
        <img src="image/logo-main.PNG" class="card-img-top" alt="find image">
       
        

        <div class="container" id="div2" >
            <input type="tel" style="display: none;" placeholder="tel" name="tel" value="<?php if(isset($_SESSION['df'])) echo $_SESSION['df']['phone'] ?>">
            <input type="password" placeholder="New Password*" name="psw1"  required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['psw1'] ?>">
            <input type="password" placeholder="Confirm Password*" name="psw2"  required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['psw2'] ?>">

            <button  type="submit">Update</button>
            
            
          </div>
      
        
      </form>

      <div class="end">
          <h6>Don't have an account yet?   <a href="sign up.html" style="margin-left: 30px">Join Us Now</a> 
              <a href="Home.html" style="margin-left: 30px">Home</a></h6>
          
          
      </div>
      


</body>
</html>
<?php
if(isset($_SESSION['errors'])){
    unset($_SESSION['errors']);
}
if(isset($_SESSION['data'])){
unset($_SESSION['data']);
}
?>