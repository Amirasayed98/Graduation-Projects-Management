<?php
    session_start();
?>
<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>log in</title>
    

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

    <form action="logindata.php" method="post">
        
        <img src="image/logo-main.PNG" class="card-img-top" alt="find image">
        <div class="container">
           <input type="text" placeholder="Accesscode for login*" name="accesscode" required value="<?php if(isset($_SESSION['accesscode'])) echo $_SESSION['data']['accesscode'] ?>">
      
          <input type="text" placeholder="Email address*" name="email" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['email'] ?>">
      
          
          <input type="password" placeholder="Password*" name="password" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['password'] ?>">
              
          <button name="submit" type="submit" onclick="document.location ='ai-cards -explore.html'">Log in</button>
          <a href="forgot-password.php" >Forgot Password</a>
          
        </div>
      
        
      </form>

      <div class="end">
          <h6>Don't have a logo account yet?   <a href="sign-up.php" style="margin-left: 30px">Join Us Now</a> 
              <a href="Home.php" style="margin-left: 30px">Home</a></h6>
          
          
      </div>


</body>
 <?php
if(isset($_SESSION['errors'])){
    unset($_SESSION['errors']);
}
if(isset($_SESSION['data'])){
    unset($_SESSION['data']);
}
?>
</html>
