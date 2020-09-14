<?php
    session_start();
?>
<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Forgot Password</title>
    

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

    <form action="Forgot_data.php" method="post">
        
        <img src="image/logo-main.PNG" class="card-img-top" alt="find image">
        <div class="container" id="div1">
          
          <input type="text" placeholder="Phone Number*" name="phone" id="num" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['phone'] ?>">
              
          <button  name="submit" type="submit"  id="">Next</button>
          
          
        </div>
        

        
        
      </form>

      <div class="end">
          <h6>Don't have an account yet?   <a href="sign-up.php" style="margin-left: 30px">Join Us Now</a> 
              <a href="Home.php" style="margin-left: 30px">Home</a></h6>
          
          
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